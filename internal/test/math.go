package test

import (
	"math"
	"github.com/pehringer/gobed/internal/vector"
)

func CosineSimilarity(left, right []float32) float32 {
	n := min(len(left), len(right))
	s := make([]float32, n)
	vector.Multiply(left, right, s)
	vector.Summation(s)
	d := s[0]
	vector.Multiply(left, left, s)
	vector.Summation(s)
	p := float32(math.Sqrt(float64(s[0])))
	vector.Multiply(right, right, s)
	vector.Summation(s)
	p *= float32(math.Sqrt(float64(s[0])))
	return d / p
}
