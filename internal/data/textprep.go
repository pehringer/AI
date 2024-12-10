package data

import (
	"bufio"
	"io"
	"strings"
	"unicode"
)

func HotEncodings(number int) [][]float32 {
	result := make([][]float32, number)
	for i := 0; i < number; i++ {
		result[i] = make([]float32, number)
		result[i][i] = 1.0
	}
	return result
}

func TokenizeText(text *bufio.Reader) (map[string]int, []int) {
	counter := 0
	lookup := map[string]int{}
	tokens := []int{}
	token := strings.Builder{}
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
			key := token.String()
			if value, present := lookup[key]; present {
				tokens = append(tokens, value)
			} else {
				tokens = append(tokens, counter)
				lookup[key] = counter
				counter++
			}
			token.Reset()
		}
		if err != nil {
			break
		}
	}
	return lookup, tokens
}
