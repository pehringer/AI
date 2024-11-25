package vec

import (
	"math"

	"github.com/pehringer/simd"
)

func Abs(a, b []float32) {
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		b[i] = float32(math.Abs(float64(a[i])))
	}
}

func Add(a, b, c []float32) {
	simd.AddFloat32(a, b, c)
}

func Dot(a, b []float32, c *float32) {
	n := min(len(a), len(b))
	d := make([]float32, n)
	simd.MulFloat32(a, b, d)
	i := n
	j := n / 2
	for j != 0 {
		simd.AddFloat32(d[0:j], d[j:i], d[0:j])
		if i%2 == 1 {
			d[0] += d[i-1]
		}
		i = j
		j /= 2
	}
	*c = d[0]
}

func ReLU(a, b []float32) {
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		b[i] = max(0.0, a[i])
	}
}

func Sub(a, b, c []float32) {
	simd.SubFloat32(a, b, c)
}
