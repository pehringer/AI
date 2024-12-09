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
	for e := 0; e < epochs; e++ {
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
	sampleIndex := 0
	batches := len(training) / batchSize
	for epoch := 0; epoch < epochs; epoch++ {
		for batch := 0; batch < batches; batch++ {
			samples := sync.WaitGroup{}
			for batchIndex := 0; batchIndex < batchSize; batchIndex++ {
				samples.Add(1)
				go func(batchIndex, sampleIndex int) {
					defer samples.Done()
					c[batchIndex].computeActivations(n.parameters, training[sampleIndex].Features, a[batchIndex])
					c[batchIndex].computeDeltas(n.parameters, a[batchIndex], training[sampleIndex].Targets, d[batchIndex])
					c[batchIndex].computeGradients(n.parameters, training[sampleIndex].Features, a[batchIndex], d[batchIndex], g[batchIndex])
				}(batchIndex, sampleIndex)
				sampleIndex += 1
				sampleIndex %= len(training)
			}
			samples.Wait()
			c[0].averageGradients(n.parameters, g)
			c[0].updateBiases(g[0], learningRate, n.parameters)
			c[0].updateWeights(g[0], learningRate, n.parameters)
		}
	}
}
