package vec

import (
	"math"

	"github.com/pehringer/simd"
)

func Add(leftOperands, rightOperands, results []float32) {
	simd.AddFloat32(leftOperands, rightOperands, results)
}

func Duplicate(operand float32, results []float32) {
	n := len(results)
	for i := 0; i < n; i++ {
		results[i] = operand
	}
}

func Multiply(leftOperands, rightOperands, results []float32) {
	simd.MulFloat32(leftOperands, rightOperands, results)
}

func ReLU(operands, results []float32) {
	n := min(len(operands), len(results))
	for i := 0; i < n; i++ {
		if operands[i] > 0 {
			results[i] = operands[i]
		} else {
			results[i] = 0.0
		}
	}
}

func ReLUDerivative(operands, results []float32) {
	n := min(len(operands), len(results))
	for i := 0; i < n; i++ {
		if operands[i] > 0 {
			results[i] = 1.0
		} else {
			results[i] = 0.0
		}
	}
}

func Softmax(operands, results []float32) {
	n := min(len(operands), len(results))
	x := operands[0]
	for i := 0; i < n; i++ {
		if operands[i] > x {
			x = operands[i]
		}
	}
	s := float32(0)
	for i := 0; i < n; i++ {
		results[i] = float32(math.Exp(float64(operands[i] - x)))
		s += results[i]
	}
	for i := 0; i < n; i++ {
		results[i] /= s
	}
}

func Subtract(leftOperands, rightOperands, results []float32) {
	simd.SubFloat32(leftOperands, rightOperands, results)
}

func Summation(result []float32) {
	i := len(result)
	j := len(result) / 2
	for j != 0 {
		simd.AddFloat32(result[0:j], result[j:i], result[0:j])
		if i%2 == 1 {
			result[0] += result[i-1]
		}
		i = j
		j /= 2
	}
}
