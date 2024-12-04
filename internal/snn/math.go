package snn

import (
	"math/rand"

	"github.com/pehringer/gobed/internal/mat"
	"github.com/pehringer/gobed/internal/vec"
)

type (
	parameters struct {
		x  int         // Input layer width.
		h  int         // Hidden layer width.
		y  int         // Output layer width.
		hb []float32   // Hidden layer biases (size h).
		hw [][]float32 // Hidden layer weights (size h by x).
		yb []float32   // Output layer biases (size y).
		yw [][]float32 // Output layer weigths (size y by h).
	}
	cache struct {
		x []float32
		h []float32
		y []float32
	}
	activations struct {
		ha []float32 // Hidden layer activations (size h).
		hz []float32 // Hidden layer inputs (size h).
		ya []float32 // Output layer activations (size y).
		yz []float32 // Output layer inputs (size y).
	}
	deltas struct {
		hd []float32 // Hidden layer error deltas (size h).
		yd []float32 // Output layer error deltas (size y).
	}
	gradients struct {
		hgb []float32   // Hidden layer bias gradients (size h).
		hgw [][]float32 // Hidden layer weight gradients (size h by x).
		ygb []float32   // Output layer bias gradients (size y).
		ygw [][]float32 // Output layer weight gradients (size y by h).
	}
)

func newParameters(x, h, y int) parameters {
	p := parameters{
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

func (p parameters) newCache() cache {
	c := cache{
		x: make([]float32, p.x),
		h: make([]float32, p.h),
		y: make([]float32, p.y),
	}
	return c
}

func (p parameters) newActivations() activations {
	a := activations{
		ha: make([]float32, p.h),
		hz: make([]float32, p.h),
		ya: make([]float32, p.y),
		yz: make([]float32, p.y),
	}
	return a
}

func (p parameters) newDeltas() deltas {
	d := deltas{
		hd: make([]float32, p.h),
		yd: make([]float32, p.y),
	}
	return d
}

func (p parameters) newGradients() gradients {
	g := gradients{
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
//	   n
// z_j = sigma(w_ij * a_i) + b_j
//	  i=0
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
func (c cache) computeActivations(p parameters, x []float32, a activations) {
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
//	   n
// d_i = sigma(w_ij  * d_j) * f'(z_i)
//	  j=0
//
// Where:
//
// d_i  Delta of the i-th neuron in the current layer.
//
// w_ij Weight from the i-th neuron in the current layer to the j-th neuron in next layer.
//
// d_j  Delta of the j-th neuron in the next layer.
//
// f'   Derivative of activation function (ReLU' or Softmax')
//
// z_i  Sum of the i-th neurons in the current layer.
func (c cache) computeDeltas(p parameters, a activations, t []float32, d deltas) {
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
func (c cache) computeGradients(p parameters, x []float32, a activations, d deltas, g gradients) {
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

func (c cache) averageGradients(p parameters, g []gradients) {
	for i := 1; i < len(g); i++ {
		vec.Add(g[0].hgb, g[i].hgb, g[0].hgb)
	}
	vec.Duplicate(float32(len(g)), c.h)
	vec.Divide(g[0].hgb, c.h, g[0].hgb)
	for i := 0; i < p.h; i++ {
		for j := 1; j < len(g); j++ {
			vec.Add(g[0].hgw[i], g[j].hgw[i], g[0].hgw[i])
		}
		vec.Duplicate(float32(len(g)), c.x)
		vec.Divide(g[0].hgw[i], c.x, g[0].hgw[i])
	}
	for i := 1; i < len(g); i++ {
		vec.Add(g[0].ygb, g[i].ygb, g[0].ygb)
	}
	vec.Duplicate(float32(len(g)), c.y)
	vec.Divide(g[0].ygb, c.y, g[0].ygb)
	for i := 0; i < p.y; i++ {
		for j := 1; j < len(g); j++ {
			vec.Add(g[0].ygw[i], g[j].ygw[i], g[0].ygw[i])
		}
		vec.Duplicate(float32(len(g)), c.h)
		vec.Divide(g[0].ygw[i], c.h, g[0].ygw[i])
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
func (c cache) updateBiases(g gradients, lr float32, p parameters) {
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
func (c cache) updateWeights(g gradients, lr float32, p parameters) {
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
