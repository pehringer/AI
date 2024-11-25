package ssn

import (
	"github.com/pehringer/gobed/internal/vec"
)

type (
	SSN struct {
		a [][]float32
		b [][]float32
		d [][]float32
		w [][][]float32
		z [][]float32
	}
)

func New(x, h, y int) SSN {
	n := SSN{
		a: [][]float32{
			make([]float32, h),
			make([]float32, y),
		},
		b: [][]float32{
			make([]float32, h),
			make([]float32, y),
		},
		d: [][]float32{
			make([]float32, h),
			make([]float32, y),
		},
		w: [][][]float32{
			make([][]float32, h),
			make([][]float32, y),
		},
		z: [][]float32{
			make([]float32, h),
			make([]float32, y),
		},
	}
	for i := 0; i < h; i++ {
		n.w[0][i] = make([]float32, x)
	}
	for i := 0; i < y; i++ {
		n.w[1][i] = make([]float32, h)
	}
	return n
}

func (n *SSN) Prediction(x []float32) []float32 {
	for i := 0; i < len(n.w[0]); i++ {
		vec.Dot(x, n.w[0][i], &n.z[0][i])
	}
	vec.Add(n.z[0], n.b[0], n.z[0])
	vec.ReLU(n.z[0], n.a[0])
	for i := 0; i < len(n.w[1]); i++ {
		vec.Dot(n.a[0], n.w[1][i], &n.z[1][i])
	}
	vec.Add(n.z[1], n.b[1], n.z[1])
	vec.Softmax(n.z[1], n.a[1])
	return n.a[1]
}
