package data

import (
	"bufio"
)

func appendSkipgrams(s []Sample, encodings [][]float32, context []int, target int) []Sample {
	for _, token := range context {
		s = append(s, Sample{
			Features: encodings[target],
			Targets:  encodings[token],
		})
	}
	return s
}

func Skipgram(text *bufio.Reader, window int) ([]Sample, map[string]int) {
	tokens, vocabulary := TokenizeText(text)
	encodings := HotEncodings(len(vocabulary))
	samples := []Sample{}
	for middle := 0; middle < len(tokens); middle++ {
		start := max(0, middle-window)
		end := min(len(tokens), middle+window+1)
		context := []int{}
		context = append(context, tokens[start:middle]...)
		context = append(context, tokens[middle+1:end]...)
		target := tokens[middle]
		samples = appendSkipgrams(samples, encodings, context, target)
	}
	return samples, vocabulary
}
