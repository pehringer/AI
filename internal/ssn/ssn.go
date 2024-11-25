package ssn

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
