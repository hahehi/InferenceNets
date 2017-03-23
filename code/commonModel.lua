require 'nngraph'
require 'cunn'

module("commonModel", package.seeall)


--nonlinear type
local function nonlinear(modelid)
	if modelid % 2 == 1 then return nn.Tanh()
	else return nn.ReLU() end
end


--encoder1
function genencoder1(types, e1dim)

	local inputs = {}
	for i = 1, #types do
		table.insert(inputs, nn.Identity()())
		table.insert(inputs, nn.Identity()())
	end
	local outputs = {}
	for i = 1, #types do
		local lookup = inputs[i] - nn.LookupTable(types[i], e1dim[i])
		local value = nn.CMulTable()({lookup, inputs[i + #types]})
		table.insert(outputs, value)
	end

	return nn.gModule(inputs, outputs)

end


--encoder2
function genencoder2(types, e1dim, e2dim, modelid)

	local inputs = {}
	local values = {}
	for i = 1, #types do
		table.insert(inputs, nn.Identity()())
		if e1dim == e2dim then
			values[i] = inputs[i]- nonlinear(modelid)
		else
			values[i] = inputs[i]- nonlinear(modelid) - nn.Linear(e1dim[i], e2dim[i]) - nonlinear(modelid)
		end
	end
	local outputs = {}
	for i = 1, #types do
		table.insert(outputs, values[i])
	end

	return nn.gModule(inputs, outputs)

end


--outcoder
function gendecoder(types, e1dim, e2dim, modelid)
		
	local inputs = {}
	local values = {}
	for i = 1, #types do
		table.insert(inputs, nn.Identity()())
		if e1dim == e2dim then
			values[i] = inputs[i] - nn.Linear(e1dim[i], types[i]) - nn.LogSoftMax()
		else
			values[i] = inputs[i] - nn.Linear(e2dim[i], e1dim[i]) - nonlinear(modelid) - nn.Linear(e1dim[i], types[i]) - nn.LogSoftMax()
		end
	end
	local outputs = {}
	for i = 1, #types do
		table.insert(outputs, values[i])
	end

	return nn.gModule(inputs, outputs)

end


--structure
local function oneinferone(id, d1, d2, d3)

	local l1
	if id <= 2 then
		l1 = nn.Sequential()
		local c1 = nn.ConcatTable()
		c1:add(nn.Sequential():add(nonlinear(id)):add(nn.Linear(d1, d3)))
		c1:add(nn.Sequential():add(nn.Linear(d1, d3)))
		l1:add(c1):add(nn.CAddTable())
	elseif id <= 4 then
		l1 = nn.Sequential():add(nn.Linear(d1, d2)):add(nonlinear(id)):add(nn.Linear(d2, d3))
	elseif id <= 6 then
		l1 = nn.Sequential()
		local c1 = nn.ConcatTable()
		c1:add(nn.Sequential():add(nn.Linear(d1, d2)):add(nonlinear(id)):add(nn.Linear(d2, d3)))
		c1:add(nn.Sequential():add(nn.Linear(d1, d3)))
		l1:add(c1):add(nn.CAddTable())
	end
	return l1

end


--all variables (incl or decl itself) infer one variable
local function allinferone(id, encodevalues, values, e2dim, mdim, outputidx, residual, residualalpha, activatefirst)

	local inputs = {}
	local addout = nn.ParallelTable()
	for i = 1, #values do
		local onetoone
		if i ~= outputidx then
			onetoone = oneinferone(id, e2dim[i], mdim[i], e2dim[outputidx])
			if activatefirst then onetoone:add(nonlinear(id)) end
			addout:add(onetoone)
			table.insert(inputs, values[i])
		else
			local alpha = oneinferone(id, e2dim[i], mdim[i], e2dim[outputidx])
			alpha:add(nn.MulConstant(residualalpha))
			if activatefirst then alpha:add(nonlinear(id)) end
			if residual then
				onetoone = nn.ConcatTable():add(alpha):add(nn.Identity())
				onetoone = nn.Sequential():add(onetoone):add(nn.CAddTable())
			else
				onetoone = nn.Sequential():add(alpha)
			end
			addout:add(onetoone)
			table.insert(inputs, values[i])
		end
	end
	local addoutpre = nn.ParallelTable()
	for i = 1, #values do
		local onetoone
		if i ~= outputidx then
			onetoone = nn.Sequential():add(nn.Linear(e2dim[i], e2dim[outputidx])):add(nn.MulConstant(0))
			addoutpre:add(onetoone)
		else
			onetoone = nn.Sequential():add(nn.Linear(e2dim[i], e2dim[outputidx]))
			if activatefirst then onetoone:add(nonlinear(id)) end
			addoutpre:add(onetoone)
		end
	end
	local mout = nn.Sequential():add(addout):add(nn.CAddTable())
	if activatefirst == false then mout:add(nonlinear(id)) end
	local moutpre = nn.Sequential():add(addoutpre):add(nn.CAddTable())
	if activatefirst == false then moutpre:add(nonlinear(id)) end
	local theout = mout(inputs)
	local theoutpre = moutpre(encodevalues)
	local madd = {theout, theoutpre} - nn.CAddTable()
	return mout, moutpre, madd

end


--all variables infer all variables, say a layer
local function allinferall(id, encodevalues, values, e2dim, mdim, residual, residualalpha, activatefirst)

	local subinfer = {}
	local subobs = {}
	local newvalues = {}
	for i = 1, #values do newvalues[i] = values[i] end
	for i = 1, #values do
		subinfer[i], subobs[i], newvalues[i] = allinferone(id, encodevalues, values, e2dim, mdim, i, residual, residualalpha, activatefirst)
	end
	return newvalues, subinfer, subobs

end


--generate graph
function gengraph(types, e1dim, e2dim, mdim, paramState)
	
	local modelid = paramState.modelid
	local recurrent = paramState.recurrent
	local residual = paramState.residual
	local residualalpha = paramState.residualalpha
	local activatefirst = paramState.activatefirst
	local infershared = paramState.infershared
	local obsshared = paramState.obsshared

	local inputs = {}
	local values = {}
	for i = 1, #types do
		table.insert(inputs, nn.Identity()())
		values[i] = inputs[i]
	end

	local subinfershare, subinfertmp, subobsshare, subobstmp
	local encodevalues = {}
	for i = 1, #values do encodevalues[i] = values[i] end
	for r = 1, recurrent do
		values, subinfertmp, subobstmp = allinferall(modelid, encodevalues, values, e2dim, mdim, residual, residualalpha, activatefirst)
		if r == 1 then
			subinfershare = subinfertmp
			subobsshare = subobstmp
		else
			if infershared == true then
				for i = 1, #values do
					subinfertmp[i]:share(subinfershare[i], 'weight', 'bias', 'gradWeight', 'gradBias')
				end
			end
			if obsshared == true then
				for i = 1, #values do
					subobstmp[i]:share(subobsshare[i], 'weight', 'bias', 'gradWeight', 'gradBias')
				end
			end
		end
	end

	local outputs = {}
	for i = 1, #types do
		outputs[i] = values[i]
	end

	return nn.gModule(inputs, outputs)

end


--generate the whole graph
function genwhole(types, e1dim, e2dim, mdim, paramState, ofname, doneEpoch, factors)

	local g, gencoder1, gencoder2, gdecoder

	if doneEpoch == 0 then
		gencoder1 = genencoder1(types, e1dim)
		gencoder2 = genencoder2(types, e1dim, e2dim, paramState.modelid)
		g = gengraph(types, e1dim, e2dim, mdim, paramState)
		gdecoder = gendecoder(types, e1dim, e2dim, paramState.modelid)
	else
		gencoder1 = torch.load(ofname.."encoder1.t7")
		gencoder2 = torch.load(ofname.."encoder2.t7")
		g = torch.load(ofname.."model.t7")
		gdecoder = torch.load(ofname.."decoder.t7")
	end

	return g, gencoder1, gencoder2, gdecoder

end