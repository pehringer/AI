package data

import (
	"bufio"
	"io"
	"strings"
	"unicode"
)

func HotEncodings(number int) [][]float32 {
	result := make([][]float32, number)
	for index := 0; index < number; index++ {
		result[index] = make([]float32, number)
		result[index][index] = 1.0
	}
	return result
}

func TokenizeText(text *bufio.Reader) (Tokens, Vocabulary) {
	counter := 0
	tokens := Tokens{}
	vocab := Vocabulary{}
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
			if value, present := vocab[key]; present {
				tokens = append(tokens, value)
			} else {
				tokens = append(tokens, counter)
				vocab[key] = counter
				counter++
			}
			token.Reset()
		}
		if err != nil {
			break
		}
	}
	return tokens, vocab
}
