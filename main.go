package main

import (
	"fmt"

	"github.com/pehringer/gobed/internal/snn"
)

func main() {
	/*
		and := [][][]float32{
			{{0.0, 0.0}, {0.0, 1.0}},
			{{0.0, 1.0}, {0.0, 1.0}},
			{{1.0, 0.0}, {0.0, 1.0}},
			{{1.0, 1.0}, {1.0, 0.0}},
		}
		oor := [][][]float32{
			{{0.0, 0.0}, {0.0, 1.0}},
			{{0.0, 1.0}, {1.0, 0.0}},
			{{1.0, 0.0}, {1.0, 0.0}},
			{{1.0, 1.0}, {1.0, 0.0}},
		}
	*/
	xor := [][][]float32{
		{{0.0, 0.0}, {0.0, 1.0}},
		{{0.0, 1.0}, {1.0, 0.0}},
		{{1.0, 0.0}, {1.0, 0.0}},
		{{1.0, 1.0}, {0.0, 1.0}},
	}
	t := xor
	p := snn.NewParameters(2, 4, 2)
	c := p.NewCache()
	a := p.NewActivations()
	d := p.NewDeltas()
	g := p.NewGradients()
	for i := 0; i < 4096; i++ {
		for _, e := range t {
			c.ComputeActivations(p, e[0], a)
			c.ComputeDeltas(p, a, e[1], d)
			c.ComputeGradients(p, e[0], a, d, g)
			c.UpdateBiases(g, 0.05, p)
			c.UpdateWeights(g, 0.05, p)
		}
		for _, e := range t {
			c.ComputeActivations(p, e[0], a)
			fmt.Println(e[0], e[1], a.Outputs())
		}
		fmt.Println()
	}
}
