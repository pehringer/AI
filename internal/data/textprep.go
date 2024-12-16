package data

import (
	"bufio"
	"strings"
	"unicode"
)

func hotEncodings(number int) [][]float32 {
	result := make([][]float32, number)
	for index := 0; index < number; index++ {
		result[index] = make([]float32, number)
		result[index][index] = 1.0
	}
	return result
}

func nextWord(text *bufio.Reader) string {
	word := strings.Builder{}
	for {
		if r, _, err := text.ReadRune(); err != nil {
			break
		} else if unicode.IsSpace(r) && word.Len() > 0 {
			break
		} else if unicode.IsLetter(r) {
			word.WriteRune(unicode.ToLower(r))
		}
	}
	return word.String()
}

func tokenizeText(text *bufio.Reader) ([]int, map[string]int) {
	counter := 0
	tokens := []int{}
	vocabulary := map[string]int{}
	for word := nextWord(text); word != ""; word = nextWord(text) {
		if _, present := vocabulary[word]; !present {
			vocabulary[word] = counter
			counter++
		}
		tokens = append(tokens, vocabulary[word])
	}
	return tokens, vocabulary
}

func GetLabels(vocabulary map[string]int) []string {
	labels := make([]string, len(vocabulary))
	for word, index := range vocabulary {
		copy := word
		labels[index] = copy
	}
	return labels
}
