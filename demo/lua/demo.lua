require 'nngraph'
require 'cunn'

--main
local starttime = os.clock()

--Load data
local ofname = "demo"
local factors = 4
local types = {11622, 55600, 32, 125}
local e1dim = {1024, 256, 32, 128}

--Load model
local g, gencoder1, gencoder2, gdecoder
gencoder1 = torch.load(ofname.."encoder1.t7")
gencoder2 = torch.load(ofname.."encoder2.t7")
g = torch.load(ofname.."model.t7")
gdecoder = torch.load(ofname.."decoder.t7")
print("defined model: "..(os.clock() - starttime))

--Define model state
local modelState = {
	gencoder1 = gencoder1,
	gencoder2 = gencoder2,
	g = g,
	gdecoder = gdecoder,
	types = types,
	e1dim = e1dim
}

function test(modelState, query, inputinds, outputinds, topn)

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
			if i ~= 3 then
				for j = 1, topn do
					table.insert(cons, confidences[1][j])
					table.insert(inds, indices[1][j])
				end
			else
				local maximum = topn
				if maximum > types[i] then maximum = types[i] end
				for j = 1, maximum do
					table.insert(cons, confidences[1][j])
					table.insert(inds, indices[1][j])
				end
				for j = maximum + 1, topn do
					table.insert(cons, 0.0)
					table.insert(inds, 1)
				end
			end
			table.insert(conss, cons)
			table.insert(indss, inds)
		end
	end

	return indss, conss

end

local query = {10263, 0, 0, 0}	--read
local topn = 10					--read

while true do

	local infile = io.open("../results/query.txt", "r")
	for i = 1, factors do
		local line = infile:read()
		query[i] = tonumber(line)
	end
	local line = infile:read()
	topn = tonumber(line)
	io.close(infile)
	
	if topn and query[1] and query[2] and query[3] and query[4] then
		local filename = "../results/"..topn.."-"..query[1].."-"..query[2].."-"..query[3].."-"..query[4]..".txt"
		local resultfile = io.open(filename)
		if not resultfile then
			local inputinds = {}
			local outputinds = {}
			for i = 1, factors do
				if query[i] == 0 then
					table.insert(outputinds, i)
				else
					table.insert(inputinds, i)
				end
			end
			local indss, conss = test(modelState, query, inputinds, outputinds, topn)
			local file = io.open(filename, "w")
			l = #indss
			for i = 1, l do
				for j = 1, topn do
					file:write(indss[i][j])
					file:write("\t")
					file:write(conss[i][j])
					file:write("\n")
				end
			end
			io.close(file)
		else
			io.close(resultfile)
		end
	end

end