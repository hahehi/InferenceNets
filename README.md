# InferenceNets

## To run the demo:

	1. cd demo
	python manage.py runserver address:port (e.g. 0.0.0.0:8000)
	2. cd demo/lua
	th demo.lua
	3. open your browser and type url "address:port/main/"

*Here some large files are not added.

### Environment requirement:

Python 2.7.* + Django 1.5.2 + lua + torch + GPU

## To use the code:

### Code description:

- commonFunc.lua: includes loading data, generating training and validation set, inference function
- commonModel.lua: generates the entire inference network
- commonTrain.lua: trains the network
- 1.lua: an example of using the code
	- commonFunc.loaddata(filename, factors)
		- filename: Data file name. Each line represents an entry tag plus an entry, which is separated by tab. The first element of each line is the entry tag and can be anything.
		- factors: The number of factors. The elements of an entry.
	- commonFunc.gettrainandtest(data, n, i)
		- n: Splits the data into n parts.
		- i: And use the i-th part to be validation set. The remained is training set.
	- paramState: You can define how many unrolling layers, whether to use identity path, share params or not, and so on.
	- commonModel.genwhole(): Generates the inference network.
	- optimState: You can define learning rate here.
	- trainingState: You can define batch size and data augmentation here.
	- commonTrain.train(): Trains the inference network.
	- commonFunc.test(): Given the query infers the result.
		- Use non-zero to indicate ids of evidence and zero to indicate unknown. E.g. query = {135, 0, 23, 0} means given the first and third factor, infer the others.

### Environment requirement:

lua + torch + GPU
