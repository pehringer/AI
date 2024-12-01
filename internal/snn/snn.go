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

// Activation Formula:
//
//         n
// z_j = sigma(w_ij * a_i) + b_j
//         i=0
//
// a_j = f(z_j)
//
// Where:
//
// z_j  Sum of the j-th neuron in the current layer.
//
// w_ij Weight from the i-th neuron in the prior layer to the j-th neuron in the current layer.
//
// a_i  Activation of the i-th neuron in the prior layer.
//
// b_j  Bias of the j-th neuron in the current layer.
//
// a_j  Activation of the j-th neuron in the current layer.
//
// f  Activation function (ReLU or Softmax)
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

// Output Layer Delta Formula:
//
// d_i = a_i  - t_i
//
// Where:
//
// d_i Delta of the i-th neuron in the current layer.
//
// a_i Activation of the i-th neuron in the current layer.
//
// t_i Expected activation of the i-th neuron in the current layer
//
// Hidden layer Delta Formula:
//
//         n
// d_i = sigma(w_ij  * d_j) * f'(z_i)
//        j=0
//
// Where:
//
// d_i  Delta of the i-th neuron in the current layer.
//
//
// w_ij Weight from the i-th neuron in the current layer to the j-th neuron in next layer.
//
//
// d_j  Delta of the j-th neuron in the next layer.
//
//
// f'   Derivative of activation function (ReLU' or Softmax')
//
// z_i  Sum of the i-th neurons in the current layer.
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

// Bias Gradient Formula:
//
// g_i = d_i
//
// Where:
//
// g_i Bias gradient of the i-th neuron in the current layer.
//
// d_i Delta of the i-th neuron in the current layer.
//
// Weight Gradient Formula:
//
// g_ij = d_j * a_i
//
// Where:
//
// g_ij weight gradient for the i-th neuron in the prior layer to the j-th neuron in the current layer.
//
// d_j  Delta of the j-th neuron in the current layer.
//
// a_i  Activation of the i-th neuron in the prior layer.
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

// Bias Update Formula:
//
// b_i = b_i - lr * g_i
//
// Where:
//
// b_i Bias of the i-th neuron in the current layer.
//
// lr  Learning rate.
//
// g_i Bias gradient of the i-th neuron in the current layer.
func (c Cache) UpdateBiases(g Gradients, lr float32, p Parameters) {
	vec.Duplicate(lr, c.h)
	vec.Multiply(c.h, g.hgb, c.h)
	vec.Subtract(p.hb, c.h, p.hb)
	vec.Duplicate(lr, c.y)
	vec.Multiply(c.y, g.ygb, c.y)
	vec.Subtract(p.yb, c.y, p.yb)
}

// Weight Update Formula:
//
// w_ij = w_ij - lr * g_ij
//
// Where:
//
// w_ij Weight from the i-th neuron in the prior layer to the j-th neuron in the current layer.
//
// lr   Learning rate.
//
// g_ij Weight gradient for the i-th neuron in the prior layer to the j-th neuron in the current layer.
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
