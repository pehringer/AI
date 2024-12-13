package data

import (
	"bufio"
	"io"
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

func tokenizeText(text *bufio.Reader) ([]int, map[string]int) {
	counter := 0
	token := strings.Builder{}
	tokens := []int{}
	vocabulary := map[string]int{}
	for {
		r, _, err := text.ReadRune()
		if err == io.EOF {
			r = ' '
		} else if err != nil {
			break
		}
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			token.WriteRune(unicode.ToLower(r))
		} else if unicode.IsSpace(r) && token.Len() > 0 {
			word := token.String()
			if value, present := vocabulary[word]; present {
				tokens = append(tokens, value)
			} else {
				tokens = append(tokens, counter)
				vocabulary[word] = counter
				counter++
			}
			token.Reset()
		}
		if err != nil {
			break
		}
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
