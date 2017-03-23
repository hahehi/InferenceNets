require 'nngraph'
require 'cunn'

module("commonFunc", package.seeall)


--load data
function loaddata(filename, dim)

	--split line
	local function split(s, delimiter)
	    local result = {}
	    for match in (s..delimiter):gmatch("(.-)"..delimiter) do
	        table.insert(result, match)
	    end
	    return result
	end

	--shuffle data
	local function shuffle(t)
		local n = #t
		while n >= 2 do
			local k = math.random(n)
			t[n], t[k] = t[k], t[n]
			n = n - 1
		end
		return t
	end

	local infile = io.open(filename)
	local data = {}
	local types = {}
	for i = 1, dim do
		types[i] = 0
	end

	--read datafile
	while true do
		local line = infile:read()
		if line == nil then break end
		local linesplit = split(line, '\t')
		local x = {}
		for i = 1, dim do
			local xi = tonumber(linesplit[i + 1]) + 1
			if xi > types[i] then types[i] = xi end
			table.insert(x, xi)
		end
		table.insert(data, x)
	end

	io.close(infile)
	--data = torch.Tensor(shuffle(data))
	data = torch.Tensor(data)

	return data, types

end


--generate trainset and testset
function gettrainandtest(data, cross, ci)

	--generate data set

	local ltest = math.floor(data:size(1) / cross)

	--get train data
	local traindata
	if ci == 1 then
		traindata = data[{{ltest*ci+1,data:size(1)},{}}]
	elseif ci == cross and ltest*ci+1 > data:size(1) then
		traindata = data[{{1,ltest*(ci-1)},{}}]
	else
		traindata = torch.cat(data[{{1,ltest*(ci-1)},{}}], data[{{ltest*ci+1,data:size(1)},{}}], 1)
	end
	--get test data
	local testdata = data[{{ltest*(ci-1)+1,ltest*ci},{}}]

	return traindata, testdata

end


--validation
function validate(modelState, testset, inputinds, outputinds, batchSize)

	local gencoder1 = modelState.gencoder1
	local gencoder2 = modelState.gencoder2
	local g = modelState.g
	local gdecoder = modelState.gdecoder
	local types = modelState.types
	local e1dim = modelState.e1dim
	local linput = #inputinds
	local loutput = #outputinds
	local factors = #types
	local inputflags = torch.Tensor(factors):fill(0)
	for i = 1, linput do
		inputflags[inputinds[i]] = i
	end

	local perplexity = {}
	for i = 1, loutput do
		table.insert(perplexity, 0)
	end
	local ltest = math.min(10000, testset.data[1]:size(1))
	local lsplits = math.floor((ltest - 1) / batchSize) + 1
	local prediction = {}

	--input
	local inputCuda = {}
	for i = 1, factors do
		table.insert(inputCuda, torch.CudaTensor(batchSize):fill(1))
	end
	for i = 1, factors do
		if inputflags[i] > 0 then
			table.insert(inputCuda, torch.CudaTensor(batchSize, e1dim[i]):fill(1))
		else
			table.insert(inputCuda, torch.CudaTensor(batchSize, e1dim[i]):fill(0))
		end
	end

	--batch based testing
	for i = 1, lsplits do

		--batch info
		local batchBegin = (i - 1) * batchSize + 1
		local batchEnd = math.min(i * batchSize, ltest)
		local batchLen = batchEnd - batchBegin + 1

		--input
		for j = 1, factors do
			if inputflags[j] > 0 then
				inputCuda[j][{{1, batchLen}}] = testset.data[inputflags[j]][{{batchBegin, batchEnd}}]
			end
		end

		--forward
		local outputEncoder = gencoder1:forward(inputCuda)
		local out = gencoder2:forward(outputEncoder)
		for i = 1, factors do
			if inputflags[i] == 0 then
				out[i]:fill(0)
			end
		end
		out = g:forward(out)
		out = gdecoder:forward(out)
		for j = 1, loutput do
			prediction[j] = out[outputinds[j]]
		end

		--perplexity
		for input = 1, loutput do
			for j = batchBegin, batchEnd do
				local predicted = prediction[input][j - batchBegin + 1]
				local value = predicted[testset.label[input][j]]
				perplexity[input] = perplexity[input] + value
			end
		end

	end

	for i = 1, loutput do
		perplexity[i] = math.exp(-perplexity[i] / ltest)
	end

	return perplexity

end


--testing
function test(modelState, query, topn)

	local gencoder1 = modelState.gencoder1
	local gencoder2 = modelState.gencoder2
	local g = modelState.g
	local gdecoder = modelState.gdecoder
	local types = modelState.types
	local e1dim = modelState.e1dim
	local inputinds = {}
	local outputinds = {}
	for i = 1, factors do
		if query[i] == 0 then
			table.insert(outputinds, i)
		else
			table.insert(inputinds, i)
		end
	end
	local linput = #inputinds
	local loutput = #outputinds
	local factors = #types
	local inputflags = torch.Tensor(factors):fill(0)
	for i = 1, linput do
		inputflags[inputinds[i]] = i
	end

	--input
	local inputCuda = {}
	for i = 1, factors do
		table.insert(inputCuda, torch.CudaTensor(1):fill(1))
	end
	for i = 1, factors do
		if inputflags[i] > 0 then
			table.insert(inputCuda, torch.CudaTensor(1, e1dim[i]):fill(1))
		else
			table.insert(inputCuda, torch.CudaTensor(1, e1dim[i]):fill(0))
		end
	end
	
	for i = 1, factors do
		if inputflags[i] > 0 then
			inputCuda[i][1] = query[i]
		end
	end

	--forward
	local outputEncoder = gencoder1:forward(inputCuda)
	local out = gencoder2:forward(outputEncoder)
	for i = 1, factors do
		if inputflags[i] == 0 then
			out[i]:fill(0)
		end
	end
	out = g:forward(out)
	out = gdecoder:forward(out)
	local conss = {}
	local indss = {}
	for i = 1, factors do
		if inputflags[i] == 0 then
			out[i]:exp()
			local confidences, indices = torch.sort(out[i], true)
			local cons = {}
			local inds = {}
			for j = 1, topn do
				table.insert(cons, confidences[1][j])
				table.insert(inds, indices[1][j])
			end
			table.insert(conss, cons)
			table.insert(indss, inds)
		end
	end

	return indss, conss

end