package snn

import (
	"runtime"
)

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
	runtime.GOMAXPROCS(runtime.NumCPU())
	c := make([]cache, batchSize)
	for i := range c {
		c[i] = n.parameters.newCache()
	}
	a := make([]activations, batchSize)
	for i := range a {
		a[i] = n.parameters.newActivations()
	}
	d := make([]deltas, batchSize)
	for i := range d {
		d[i] = n.parameters.newDeltas()
	}
	g := make([]gradients, batchSize)
	for i := range d {
		g[i] = n.parameters.newGradients()
	}
	i := 0
	batches := len(data) / batchSize
	for e := 0; e < epochs; e++ {
		for b := 0; b < batches; b++ {
			for r := 0; r < batchSize; r++ {
				c[r].computeActivations(n.parameters, data[i].Features, a[r])
				c[r].computeDeltas(n.parameters, a[r], data[i].Targets, d[r])
				c[r].computeGradients(n.parameters, data[i].Features, a[r], d[r], g[r])
				i += 1
				i %= len(data)
			}
			c[0].averageGradients(n.parameters, g)
			c[0].updateBiases(g[0], learningRate, n.parameters)
			c[0].updateWeights(g[0], learningRate, n.parameters)
		}
	}
}
