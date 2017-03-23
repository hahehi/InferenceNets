require 'nngraph'
require 'cunn'
require 'commonFunc'

module("commonTrain", package.seeall)


--shuffle training data
local function shuffletrain(data)

	local n = data:size(1)
	local d = data:size(2)
	while n >= 2 do
		local k = math.random(n)
		for i = 1, d do
			data[n][i], data[k][i] = data[k][i], data[n][i]
		end
		n = n - 1
	end
	return data

end


--4 point decimal
local function p4(x) return string.format("%.4f", x) + 0 end


--print to file
local function printtofile(ofname, tloss, vloss, perplexity, epoch)

	file = io.open("result"..ofname..".txt", "a")
	if epoch then
		file:write(epoch)
		file:write("\t")
	end
	file:write(p4(tloss))
	file:write("\t")
	file:write(p4(vloss))
	for i = 1, #perplexity do
		file:write("\t")
		file:write(p4(perplexity[i]))
	end
	file:write("\n")
	file:close()

end


--5 effective decimal
local function s5(x) return string.format("%.5g", x) + 0 end


--get batch
function getbatch(data, inputinds, outputinds, beginidx, batchSize)
	local ldata = data:size(1)
	local endidx = math.min(beginidx + batchSize - 1, ldata)
	beginidx = endidx - batchSize + 1
	local datas = {}
	local labels = {}
	for i = 1, #inputinds do
		datas[i] = torch.Tensor(batchSize)
	end
	for i = 1, #outputinds do
		labels[i] = torch.Tensor(batchSize)
	end
	for di = beginidx, endidx do
		for i = 1, #inputinds do
			datas[i][di - beginidx + 1] = data[di][inputinds[i]]
		end
		for i = 1, #outputinds do
			labels[i][di - beginidx + 1] = data[di][outputinds[i]]
		end
	end
	return {data = datas, label = labels}
end


--get inputs and outputs from idx
function getinsandouts(factors, idx)
	local inputinds = {}
	local outputinds = {}
	for i = 1, factors do
		local b = 2 ^ (i - 1)
		local v = math.floor(idx / b) % 2
		if v == 1 then
			table.insert(inputinds, i)
		else
			table.insert(outputinds, i)
		end
	end
	return inputinds, outputinds
end


--generate data noise
local function gennoise(data, types, i, ltraindata)
	--calcu variables distribution
	local counter = torch.Tensor(types[i]):fill(0)
	for j = 1, ltraindata do
		local value = data[i][j]
		counter[value] = counter[value] + 1
	end
	counter = counter / ltraindata
	local confidences, indices = torch.sort(counter, true)
	for j = 2, confidences:size(1) do
		confidences[j] = confidences[j] + confidences[j - 1]
	end

	local randi = torch.rand(ltraindata)
	for j = 1, ltraindata do
		local idx = 1
		for k = 1, types[i] do
			if confidences[k] >= randi[j] then
				idx = indices[k]
				break
			end
		end
		randi[j] = idx
	end
	return randi
end


--test all
local function testall(modelState, inouttypes, factors, traindata, testdata, batchSize, ofname, epoch, tlosstotal, tlosseach)
	local vlosstotal = 0
	local vlosseach = {}
	local perplexityeach = {}
	if epoch % 5 == 0 then tlosstotal = 0 end
	for idx = 1, inouttypes do
		local inputinds, outputinds = getinsandouts(factors, idx)
		local testset = getbatch(testdata, inputinds, outputinds, 1, testdata:size(1))
		local perplexity = commonFunc.validate(modelState, testset, inputinds, outputinds, batchSize)

		--print testing result
		local vloss = 0
		for i = 1, #outputinds do
			vloss = vloss + math.log(perplexity[i])
		end
		local allpplxt = {}
		for i = 1, factors do table.insert(allpplxt, 0) end
		for i = 1, #outputinds do allpplxt[outputinds[i]] = perplexity[i] end

		if epoch % 5 == 0 then
			local testset = getbatch(traindata, inputinds, outputinds, 1, traindata:size(1))
			local perplexity = commonFunc.validate(modelState, testset, inputinds, outputinds, batchSize)
			local tloss = 0
			for i = 1, #outputinds do
				tloss = tloss + math.log(perplexity[i])
			end
			tlosstotal = tlosstotal + tloss
			tlosseach[idx] = tloss
		end

		print("-tloss:", s5(tlosseach[idx]), "vloss:", s5(vloss), "pplxt:", s5(allpplxt[1]), s5(allpplxt[2]), s5(allpplxt[3]), s5(allpplxt[4]))
		printtofile(ofname, tlosseach[idx], vloss, allpplxt)
		vlosstotal = vlosstotal + vloss
		table.insert(vlosseach, vloss)
		table.insert(perplexityeach, allpplxt)
	end
	return vlosstotal, vlosseach, perplexityeach, tlosstotal, tlosseach
end


--training
function train(trainingState, testState, modelState)

	--training state
	local traindata = trainingState.traindata
	local optimState = trainingState.optimState
	local beginEpoch = trainingState.beginEpoch
	local totalEpoch = trainingState.totalEpoch
	local batchSize = trainingState.batchSize
	local augnum = trainingState.augnum
	local augmax = trainingState.augmax
	local addnoise = trainingState.addnoise

	--test state
	local testdata = testState.testdata
	local ofname = testState.ofname

	--model state
	local shared = modelState.shared
	local gencoder1 = modelState.gencoder1
	local gencoder2 = modelState.gencoder2
	local g = modelState.g
	local gdecoder = modelState.gdecoder
	local types = modelState.types
	local e1dim = modelState.e1dim
	local e2dim = modelState.e2dim

	--some useful marks
	local factors = #types
	local inouttypes = 2 ^ factors - 2
	local ltraindata = traindata:size(1)
	local augSize = batchSize * augnum
	local lowestepoch = beginEpoch - 1
	local lowestlosseach = {}
	local lowestloss = 0
	local tlosseach = {}
	for i = 1, inouttypes do
		local inputinds, outputinds = getinsandouts(factors, i)
		local loss = 0
		for j = 1, #outputinds do loss = loss + math.log(types[outputinds[j]]) end
		table.insert(lowestlosseach, loss)
		table.insert(tlosseach, loss)
		lowestloss = lowestloss + lowestlosseach[i]
	end
	local tlosstotal = lowestloss

	--to cuda
	gencoder1 = gencoder1:cuda()
	gencoder2 = gencoder2:cuda()
	g = g:cuda()
	gdecoder = gdecoder:cuda()
	modelState.gencoder1 = gencoder1
	modelState.gencoder2 = gencoder2
	modelState.g = g
	modelState.gdecoder = gdecoder

	--params
	local e1params, e1gradParams = gencoder1:getParameters()
	local e2params, e2gradParams = gencoder2:getParameters()
	local params, gradParams = g:getParameters()
	local dparams, dgradParams = gdecoder:getParameters()

	--data input and output
	local inputCuda = {}
	local outputCuda = {}
	for i = 1, factors do
		table.insert(inputCuda, torch.CudaTensor(batchSize):fill(1))
	end
	for i = 1, factors do
		table.insert(inputCuda, torch.CudaTensor(batchSize, e1dim[i]))
	end
	for i = 1, factors do
		table.insert(outputCuda, torch.CudaTensor(batchSize):fill(1))
	end

	--data noise
	local auginputCuda = {}
	local augdata = {}
	if augnum > 0 then
		for i = 1, factors do
			table.insert(auginputCuda, torch.CudaTensor(augSize):fill(1))
		end
		for i = 1, factors do
			table.insert(auginputCuda, torch.CudaTensor(augSize, e1dim[i]))
		end
		for i = 1, factors do
			if io.open("datarand"..i..'.t7') then
				augdata[i] = torch.load("datarand"..i..'.t7')
			else
				augdata[i] = gennoise(data, types, i, ltraindata)
				torch.save("datarand"..i..'.t7', augdata[i])
			end
		end
	end

	--data encode
	local encodeCuda = {}
	for i = 1, factors do
		local encode = torch.CudaTensor(batchSize, e1dim[i])
		table.insert(encodeCuda, encode)
	end


	--batch based training procedure
	local function trainbatch(trainset, inputinds, outputinds)

		local linput = #inputinds
		local loutput = #outputinds
		local inputflags = torch.Tensor(factors):fill(0)
		local outputflags = torch.Tensor(factors):fill(0)
		for i = 1, linput do
			inputflags[inputinds[i]] = i
		end
		for i = 1, loutput do
			outputflags[outputinds[i]] = i
		end
		local criterion = nn.ParallelCriterion()
		for i = 1, factors do
			if outputflags[i] == 0 then
				criterion:add(nn.ClassNLLCriterion(), 0)
			else
				criterion:add(nn.ClassNLLCriterion())
			end
		end
		criterion = criterion:cuda()
		
		e1gradParams:zero()
		e2gradParams:zero()
		gradParams:zero()
		dgradParams:zero()

		--input and output
		for i = 1, factors do
			if inputflags[i] > 0 then
				inputCuda[i][{{}}] = trainset.data[inputflags[i]][{{}}]
				inputCuda[i + factors]:fill(1)
			else
				outputCuda[i][{{}}] = trainset.label[outputflags[i]][{{}}]
				inputCuda[i + factors]:fill(0)
			end
		end

		--noise
		if augnum > 0 then
			for i = 1, factors do
				auginputCuda[factors + i]:fill(0)
				if inputflags[i] > 0 then
					local beginfetch = torch.random(1, ltraindata - augSize + 1)
					auginputCuda[i]:copy(augdata[i][{{beginfetch, beginfetch + augSize - 1}}])
					if addnoise[i] then auginputCuda[factors + i]:fill(augmax) end
				end
			end
		end

		--forward gencoder1
		local outputEncode = gencoder1:forward(inputCuda)
		for i = 1, factors do
			encodeCuda[i]:copy(outputEncode[i])
		end
		outputEncode = encodeCuda
		for j = 1, augnum do
			local beginidx = (j - 1) * batchSize + 1
			local endidx = j * batchSize
			local tmpinput = {}
			for i = 1, 2 * factors do table.insert(tmpinput, auginputCuda[i][{{beginidx, endidx}}]) end
			local augoutputEncode = gencoder1:forward(tmpinput)
			for i = 1, factors do
				outputEncode[i]:add(augoutputEncode[i])
			end
		end

		--forward gencoder2, g, gdecoder
		local inferinputs = gencoder2:forward(outputEncode)
		for i = 1, factors do
			if inputflags[i] == 0 then
				inferinputs[i]:fill(0)
			end
		end
		local inferoutputs = g:forward(inferinputs)
		local totaloutputs = gdecoder:forward(inferoutputs)

		--forward crit
		local loss = criterion:forward(totaloutputs, outputCuda)

		--backward crit
		local gloss = criterion:backward(totaloutputs, outputCuda)

		--backward gdecoder, g, gencoder2
		local goutput = gdecoder:backward(inferoutputs, gloss)
		goutput = g:backward(inferinputs, goutput)
		for i = 1, factors do
			if inputflags[i] == 0 then
				goutput[i]:fill(0)
			end
		end
		goutput = gencoder2:backward(outputEncode, goutput)

		--backward gencoder
		for j = augnum, 1, -1 do
			local beginidx = (j - 1) * batchSize + 1
			local endidx = j * batchSize
			local tmpinput = {}
			for i = 1, 2 * factors do table.insert(tmpinput, auginputCuda[i][{{beginidx, endidx}}]) end
			gencoder1:backward(tmpinput, goutput)
		end
		gencoder1:backward(inputCuda, goutput)

		--update params
		gencoder1:updateParameters(optimState.learningRate)
		gencoder2:updateParameters(optimState.learningRate)
		g:updateParameters(optimState.learningRate)
		gdecoder:updateParameters(optimState.learningRate)

		return loss

	end


	------------------------------------------
	------------------------------------------
	--------------begin training--------------
	------------------------------------------
	------------------------------------------

	local epoch = beginEpoch

	--for each epoch
	while epoch <= totalEpoch do

		--shuffle training data
		local starttime = os.clock()
		traindata = shuffletrain(traindata)

		--for each batch
		for batchBegin = 1, ltraindata, batchSize do

			--get batch and train batch
			local batchEnd = math.min(batchBegin + batchSize - 1, ltraindata)
			local randtypes = math.random(1, inouttypes)
			local inputinds, outputinds = getinsandouts(factors, randtypes)
			local trainset = getbatch(traindata, inputinds, outputinds, batchBegin, batchSize)
			local loss = trainbatch(trainset, inputinds, outputinds)

			--the end of an epoch
			if batchEnd == ltraindata then

				local duringtime = os.clock() - starttime
				starttime = os.clock()

				--print testing start
				print("epoch:", epoch, "lepoch:", lowestepoch, "tloss:", s5(tlosstotal), "lloss:", s5(lowestloss), 'time:', s5(duringtime))
				
				--test
				local vlosstotal, vlosseach, perplexityeach
				vlosstotal, vlosseach, perplexityeach, tlosstotal, tlosseach = testall(modelState, inouttypes, factors, traindata, testdata, batchSize, ofname, epoch, tlosstotal, tlosseach)
				duringtime = os.clock() - starttime
				starttime = os.clock()
				if batchEnd == ltraindata then print('testing time:', s5(duringtime)) end

				--get new record! save and print it
				if vlosstotal < lowestloss then

					torch.save(ofname.."encoder1.t7", gencoder1:clearState())
					torch.save(ofname.."encoder2.t7", gencoder2:clearState())
					torch.save(ofname.."model.t7", g:clearState())
					torch.save(ofname.."decoder.t7", gdecoder:clearState())
					torch.save(ofname.."optim.t7", optimState)
					lowestloss = vlosstotal
					lowestepoch = epoch
					for i = 1, inouttypes do
						lowestlosseach[i] = vlosseach[i]
						printtofile(ofname..'optim', tlosseach[i], vlosseach[i], perplexityeach[i], epoch)
					end

				end

			end

		end

		epoch = epoch + 1

	end

end