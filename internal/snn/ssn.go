package snn

type (
	SampleData struct {
		Features []float32
		Targets  []float32
	}
	TrainingSet   []SampleData
	NeuralNetwork struct {
		parameters
	}
)

func NewNeuralNetwork(inputWidth, hiddenWidth, outputWidth int) NeuralNetwork {
	return NeuralNetwork{
		newParameters(inputWidth, hiddenWidth, outputWidth),
	}
}

func (n NeuralNetwork) Prediction(features []float32) []float32 {
	c := n.parameters.newCache()
	a := n.parameters.newActivations()
	c.computeActivations(n.parameters, features, a)
	return a.ya
}

func (n NeuralNetwork) OnlineTrain(data TrainingSet, epochs int, learningRate float32) {
	c := n.parameters.newCache()
	a := n.parameters.newActivations()
	d := n.parameters.newDeltas()
	g := n.parameters.newGradients()
	for e := 0; e < epochs; e++ {
		for _, sample := range data {
			c.computeActivations(n.parameters, sample.Features, a)
			c.computeDeltas(n.parameters, a, sample.Targets, d)
			c.computeGradients(n.parameters, sample.Features, a, d, g)
			c.updateBiases(g, learningRate, n.parameters)
			c.updateWeights(g, learningRate, n.parameters)
		}
	}
}

func (n NeuralNetwork) BatchTrain(data TrainingSet, epochs, batchSize int, learningRate float32) {
	// TODO
}
