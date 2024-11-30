package snn

import (
	"math/rand"

	"github.com/pehringer/gobed/internal/mat"
	"github.com/pehringer/gobed/internal/vec"
)

type (
	Parameters struct {
		x  int
		h  int
		y  int
		hb []float32
		hw [][]float32
		yb []float32
		yw [][]float32
	}
	Cache struct {
		x []float32
		h []float32
		y []float32
	}
	Activations struct {
		ha []float32
		hz []float32
		ya []float32
		yz []float32
	}
	Deltas struct {
		hd []float32
		yd []float32
	}
	Gradients struct {
		hgb []float32
		hgw [][]float32
		ygb []float32
		ygw [][]float32
	}
)

func NewParameters(x, h, y int) Parameters {
	p := Parameters{
		x:  x,
		h:  h,
		y:  y,
		hb: make([]float32, h),
		hw: make([][]float32, h),
		yb: make([]float32, y),
		yw: make([][]float32, y),
	}
	for i := 0; i < h; i++ {
		p.hb[i] = rand.Float32() * 0.01
		p.hw[i] = make([]float32, x)
		for j := 0; j < x; j++ {
			p.hw[i][j] = rand.Float32() * 0.01
		}
	}
	for i := 0; i < y; i++ {
		p.yb[i] = rand.Float32() * 0.01
		p.yw[i] = make([]float32, h)
		for j := 0; j < h; j++ {
			p.yw[i][j] = rand.Float32() * 0.01
		}
	}
	return p
}

func (p Parameters) NewCache() Cache {
	c := Cache{
		x: make([]float32, p.x),
		h: make([]float32, p.h),
		y: make([]float32, p.y),
	}
	return c
}

func (p Parameters) NewActivations() Activations {
	a := Activations{
		ha: make([]float32, p.h),
		hz: make([]float32, p.h),
		ya: make([]float32, p.y),
		yz: make([]float32, p.y),
	}
	return a
}

func (p Parameters) NewDeltas() Deltas {
	d := Deltas{
		hd: make([]float32, p.h),
		yd: make([]float32, p.y),
	}
	return d
}

func (p Parameters) NewGradients() Gradients {
	g := Gradients{
		hgb: make([]float32, p.h),
		hgw: make([][]float32, p.h),
		ygb: make([]float32, p.y),
		ygw: make([][]float32, p.y),
	}
	for i := 0; i < p.h; i++ {
		g.hgw[i] = make([]float32, p.x)
	}
	for i := 0; i < p.y; i++ {
		g.ygw[i] = make([]float32, p.h)
	}
	return g
}

func (c Cache) ComputeActivations(p Parameters, x []float32, a Activations) {
	for i := 0; i < p.h; i++ {
		vec.Multiply(x, p.hw[i], c.x)
		vec.Summation(c.x)
		a.hz[i] = c.x[0]
	}
	vec.Add(a.hz, p.hb, a.hz)
	vec.ReLU(a.hz, a.ha)
	for i := 0; i < p.y; i++ {
		vec.Multiply(a.ha, p.yw[i], c.h)
		vec.Summation(c.h)
		a.yz[i] = c.h[0]
	}
	vec.Add(a.yz, p.yb, a.yz)
	vec.Softmax(a.yz, a.ya)
}

func (a Activations) Outputs() []float32 {
	return a.ya
}

func (c Cache) ComputeDeltas(p Parameters, a Activations, t []float32, d Deltas) {
	vec.Subtract(a.ya, t, d.yd)
	vec.ReLUDerivative(a.hz, d.hd)
	for i := 0; i < p.h; i++ {
		mat.Column(p.yw, i, c.y)
		vec.Multiply(c.y, d.yd, c.y)
		vec.Summation(c.y)
		c.h[i] = c.y[0]
	}
	vec.Multiply(d.hd, c.h, d.hd)
}

func (c Cache) ComputeGradients(p Parameters, x []float32, a Activations, d Deltas, g Gradients) {
	copy(g.hgb, d.hd)
	for i := 0; i < p.h; i++ {
		vec.Duplicate(d.hd[i], c.x)
		vec.Multiply(x, c.x, g.hgw[i])
	}
	copy(g.ygb, d.yd)
	for i := 0; i < p.y; i++ {
		vec.Duplicate(d.yd[i], c.h)
		vec.Multiply(a.ha, c.h, g.ygw[i])
	}
}

func (c Cache) UpdateBiases(g Gradients, lr float32, p Parameters) {
	vec.Duplicate(lr, c.h)
	vec.Multiply(c.h, g.hgb, c.h)
	vec.Subtract(p.hb, c.h, p.hb)
	vec.Duplicate(lr, c.y)
	vec.Multiply(c.y, g.ygb, c.y)
	vec.Subtract(p.yb, c.y, p.yb)
}

func (c Cache) UpdateWeights(g Gradients, lr float32, p Parameters) {
	for i := 0; i < p.h; i++ {
		vec.Duplicate(lr, c.x)
		vec.Multiply(c.x, g.hgw[i], c.x)
		vec.Subtract(p.hw[i], c.x, p.hw[i])
	}
	for i := 0; i < p.y; i++ {
		vec.Duplicate(lr, c.h)
		vec.Multiply(c.h, g.ygw[i], c.h)
		vec.Subtract(p.yw[i], c.h, p.yw[i])
	}
}
