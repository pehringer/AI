package snn

import (
	"runtime"
	"sync"

	"github.com/pehringer/gobed/internal/data"
)

type (
	NeuralNetwork struct {
		parameters
	}
)

func Initialize(inputWidth, hiddenWidth, outputWidth int) NeuralNetwork {
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

func (n NeuralNetwork) OnlineTrain(training data.Set, epochs int, learningRate float32) {
	c := n.parameters.newCache()
	a := n.parameters.newActivations()
	d := n.parameters.newDeltas()
	g := n.parameters.newGradients()
	for epoch := 0; epoch < epochs; epoch++ {
		for _, sample := range training {
			c.computeActivations(n.parameters, sample.Features, a)
			c.computeDeltas(n.parameters, a, sample.Targets, d)
			c.computeGradients(n.parameters, sample.Features, a, d, g)
			c.updateBiases(g, learningRate, n.parameters)
			c.updateWeights(g, learningRate, n.parameters)
		}
	}
}

func (n NeuralNetwork) BatchTrain(training data.Set, epochs, batchSize int, learningRate float32) {
	runtime.GOMAXPROCS(runtime.NumCPU())
	batchCaches := make([]cache, batchSize)
	for index := range batchCaches {
		batchCaches[index] = n.parameters.newCache()
	}
	batchActivations := make([]activations, batchSize)
	for index := range batchActivations {
		batchActivations[index] = n.parameters.newActivations()
	}
	batchDeltas := make([]deltas, batchSize)
	for index := range batchDeltas {
		batchDeltas[index] = n.parameters.newDeltas()
	}
	batchGradients := make([]gradients, batchSize)
	for index := range batchDeltas {
		batchGradients[index] = n.parameters.newGradients()
	}
	sampleIndex := 0
	batches := len(training) / batchSize
	for epoch := 0; epoch < epochs; epoch++ {
		for batch := 0; batch < batches; batch++ {
			samples := sync.WaitGroup{}
			for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
				samples.Add(1)
				go func(batchIndex, sampleIndex int) {
					defer samples.Done()
					batchCaches[batchIndex].computeActivations(n.parameters, training[sampleIndex].Features, batchActivations[batchIndex])
					batchCaches[batchIndex].computeDeltas(n.parameters, batchActivations[batchIndex], training[sampleIndex].Targets, batchDeltas[batchIndex])
					batchCaches[batchIndex].computeGradients(n.parameters, training[sampleIndex].Features, batchActivations[batchIndex], batchDeltas[batchIndex], batchGradients[batchIndex])
				}(batchIndex, sampleIndex)
				sampleIndex += 1
				sampleIndex %= len(training)
			}
			samples.Wait()
			batchCaches[0].averageGradients(n.parameters, batchGradients)
			batchCaches[0].updateBiases(batchGradients[0], learningRate, n.parameters)
			batchCaches[0].updateWeights(batchGradients[0], learningRate, n.parameters)
		}
	}
}
