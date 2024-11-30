package mat

func Column(a [][]float32, b int, c []float32) {
	n := len(a)
	for i := 0; i < n; i++ {
		m := len(a[i])
		if b < m {
			c[i] = a[i][b]
		}
	}
}
