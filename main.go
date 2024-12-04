package main

import (
	"fmt"

	"github.com/pehringer/gobed/internal/snn"
)

var (
	or = snn.TrainingSet{
		{
			Features: []float32{0.0, 0.0},
			Targets:  []float32{0.0, 1.0},
		},
		{
			Features: []float32{0.0, 1.0},
			Targets:  []float32{1.0, 0.0},
		},
		{
			Features: []float32{1.0, 0.0},
			Targets:  []float32{1.0, 0.0},
		},
		{
			Features: []float32{1.0, 1.0},
			Targets:  []float32{1.0, 0.0},
		},
	}
	nor = snn.TrainingSet{
		{
			Features: []float32{0.0, 0.0},
			Targets:  []float32{1.0, 0.0},
		},
		{
			Features: []float32{0.0, 1.0},
			Targets:  []float32{0.0, 1.0},
		},
		{
			Features: []float32{1.0, 0.0},
			Targets:  []float32{0.0, 1.0},
		},
		{
			Features: []float32{1.0, 1.0},
			Targets:  []float32{0.0, 1.0},
		},
	}
	xor = snn.TrainingSet{
		{
			Features: []float32{0.0, 0.0},
			Targets:  []float32{0.0, 1.0},
		},
		{
			Features: []float32{0.0, 1.0},
			Targets:  []float32{1.0, 0.0},
		},
		{
			Features: []float32{1.0, 0.0},
			Targets:  []float32{1.0, 0.0},
		},
		{
			Features: []float32{1.0, 1.0},
			Targets:  []float32{0.0, 1.0},
		},
	}
)

func main() {
	ts := or
	n := snn.NewNeuralNetwork(2, 4, 2)
	//n.OnlineTrain(ts, 4096, 0.05)
	n.BatchTrain(ts, 8192, 3, 0.05)
	for i := 0; i < len(ts); i++ {
		fmt.Println(ts[i].Features, ts[i].Targets, n.Prediction(ts[i].Features))
	}
}
