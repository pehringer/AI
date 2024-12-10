package main

import (
	"fmt"

	"github.com/pehringer/gobed/internal/data"
	"github.com/pehringer/gobed/internal/snn"
)

var (
	or = data.Set{
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
	nor = data.Set{
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
	xor = data.Set{
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
	and = data.Set{
		{
			Features: []float32{0.0, 0.0},
			Targets:  []float32{0.0, 1.0},
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
			Targets:  []float32{1.0, 0.0},
		},
	}
	nand = data.Set{
		{
			Features: []float32{0.0, 0.0},
			Targets:  []float32{1.0, 0.0},
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
	fmt.Println("Testing Hot Encodings: 4")
	for _, vec := range data.HotEncodings(4) {
		fmt.Println(vec)
	}
	fmt.Println("Training Logic Gate:")
	ts := nand
	n := snn.Initialize(2, 4, 2)
	//n.OnlineTrain(ts, 4096, 0.05)
	n.BatchTrain(ts, 25000, 3, 0.05)
	for i := 0; i < len(ts); i++ {
		fmt.Println(ts[i].Features, ts[i].Targets, n.Prediction(ts[i].Features))
	}
}
