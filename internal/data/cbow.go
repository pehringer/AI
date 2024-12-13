package data

import (
	"bufio"
)

func combineFeatures(encodings [][]float32, tokens []int) []float32 {
	features := make([]float32, len(encodings))
	for _, token := range tokens {
		for index, value := range encodings[token] {
			features[index] += value
		}
	}
	return features
}

func appendCbow(s []Sample, encodings [][]float32, context []int, target int) []Sample {
	features := combineFeatures(encodings, context)
	s = append(s, Sample{
		Features: features,
		Targets:  encodings[target],
	})
	return s
}

func Cbow(text *bufio.Reader, window int) ([]Sample, map[string]int) {
	tokens, vocabulary := tokenizeText(text)
	encodings := hotEncodings(len(vocabulary))
	samples := []Sample{}
	for middle := 0; middle < len(tokens); middle++ {
		start := max(0, middle-window)
		end := min(len(tokens), middle+window+1)
		context := append([]int{}, tokens[start:middle]...)
		context = append(context, tokens[middle+1:end]...)
		target := tokens[middle]
		samples = appendCbow(samples, encodings, context, target)
	}
	return samples, vocabulary
}
