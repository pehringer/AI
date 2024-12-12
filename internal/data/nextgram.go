package data

import (
	"bufio"
)

func CombineFeatures(encodings [][]float32, context []int) []float32 {
	features := make([]float32, len(encodings[0]))
	for _, token := range context {
		for index, value := range encodings[token] {
			features[index] += value
		}
	}
	return features
}

func appendNextgrams(s []Sample, encodings [][]float32, context []int, target int) []Sample {
	features := CombineFeatures(encodings, context)
	s = append(s, Sample{
			Features: features,
			Targets:  encodings[target],
		})
	return s
}

func Nextgram(text *bufio.Reader, window int) ([]Sample, map[string]int) {
	tokens, vocabulary := TokenizeText(text)
	encodings := HotEncodings(len(vocabulary))
	samples := []Sample{}
	for end := 0; end < len(tokens); end++ {
		start := max(0, end-window)
		context := append([]int{}, tokens[start:end]...)
		target := tokens[end]
		samples = appendSkipgrams(samples, encodings, context, target)
	}
	return samples, vocabulary
}
