require 'commonFunc'
require 'commonTrain'
require 'commonModel'


--main
local starttime = os.clock()

--Load data
local ofname = 1			--output file index
local doneEpoch = 0			--0: train from very beginning; >0: train from previous state saved
local factors = 4			--the number of factors in the data
local e1dim = {1024, 256, 32, 128}		--the first-layer encoding dimensions for all factors
local e2dim = {256, 64, 32, 64}			--the second-layer encoding dimensions for all factors
local mdim = {32, 16, 32, 32}			--the dimensions of middle representations for all factors
local data, types = commonFunc.loaddata("data-imdb.txt", factors)	--load data
local traindata, testdata = commonFunc.gettrainandtest(data, 5, 1)		--get training and validation data
print("load data: "..(os.clock() - starttime))

--Define param state
local paramState = {
	modelid = 6,			
	recurrent = 4,			--the number of unrolling layers
	singleende = true,		--true: 1-layer en/decoding; false: 2-layer
	residual = true,		--true: with identity path; false: without identity path
	residualalpha = 1.0,	
	activatefirst = false,	
	infershared = false,	--true: share inference params; false: not share
	obsshared = false		--true: share observation params; false: not share
}
if paramState.singleende then e2dim = e1dim end
local g, gencoder1, gencoder2, gdecoder = commonModel.genwhole(types, e1dim, e2dim, mdim, paramState, ofname, doneEpoch, factors)
print("define model: "..(os.clock() - starttime))

--Define optim state
local optimState = { learningRate = 0.01 }			--define initial learning rate
if doneEpoch > 0 then
	optimState = torch.load(ofname.."optim.t7")
end

--Define training state, validation state, and model state
local trainingState = {
	traindata = traindata,
	optimState = optimState,
	beginEpoch = doneEpoch + 1,
	totalEpoch = 10000,								--define the maximum epoch to train
	batchSize = 200,								--define batch size
	augnum = 2,										--define the size of data augmentation
	augmax = 0.1,									--define the noise of data augmentation
	addnoise = {true, true, true, true}				--define if using data augmentation for each factor
}
local validationState = {
	testdata = testdata,
	ofname = ofname
}
local modelState = {
	shared = paramState.shared,
	gencoder1 = gencoder1,
	gencoder2 = gencoder2,
	g = g,
	gdecoder = gdecoder,
	types = types,
	e1dim = e1dim,
	e2dim = e2dim
}

--Begin training
commonTrain.train(trainingState, validationState, modelState)	--training

--Begin testing
local query = {135, 0, 23, 0}							--given the first and third factor, infer the others
local topn = 10											--return the topn results
local indss, conss = commonFunc.test(modelState, query, topn)	--testing
local file = io.open("result.txt", "w")
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