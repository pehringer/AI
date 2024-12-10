package data

func HotEncodings(number int) [][]float32 {
	result := make([][]float32, number)
	for i := 0; i < number; i++ {
		result[i] = make([]float32, number)
		result[i][i] = 1.0
	}
	return result
}

func tokenizeText(text string) (map[string]int, []int) {
	return nil, nil
}
