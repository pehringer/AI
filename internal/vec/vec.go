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


func Sum(a []float32) {
	i := len(a)
	j := len(a) / 2
	for j != 0 {
		simd.AddFloat32(a[0:j], a[j:i], a[0:j])
		if i%2 == 1 {
			a[0] += a[i-1]
		}
		i = j
		j /= 2
	}
}

func Dot(a, b []float32, c *float32) {
	n := min(len(a), len(b))
	d := make([]float32, n)
	simd.MulFloat32(a, b, d)
	Sum(d)
	*c = d[0]
}

func ReLU(a, b []float32) {
	n := min(len(a), len(b))
	for i := 0; i < n; i++ {
		b[i] = float32(math.Max(0.0, float64(a[i])))
	}
}

func Softmax(a, b []float32) {
    n := min(len(a), len(b))
    m := a[0]
    for i := 0; i < n; i++ {
        if a[i] > m {
            m = a[i]
        }
    }
    s := float32(0)
    for i := 0; i < n; i++ {
        b[i] = float32(math.Exp(float64(a[i] - m)))
        s += b[i]
    }
    for i := 0; i < n; i++ {
        b[i] /= s
    }
}

func Sub(a, b, c []float32) {
	simd.SubFloat32(a, b, c)
}
