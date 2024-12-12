package main

import (
	"bufio"
	"fmt"
	"os"

	"github.com/pehringer/gobed/internal/data"
	"github.com/pehringer/gobed/internal/snn"
)

func main() {
	file, _ := os.Open("text/TheForgottenVillage.txt")
	reader := bufio.NewReader(file)

	trainingSamples, vocabulary := data.Nextgram(reader, 3)
	fmt.Println(len(trainingSamples), len(vocabulary))
	network := snn.Initialize(len(vocabulary), 32, len(vocabulary))
	network.BatchTrain(trainingSamples, 150, 16, 0.10)

	file, _ = os.Open("text/TheForgottenVillage.txt")
	reader = bufio.NewReader(file)
	tokens, _ := data.TokenizeText(reader)
	labels := data.GetLabels(vocabulary)
	encodings := data.HotEncodings(len(vocabulary))
	for i := 3; i < len(tokens); i++ {
		inputs := data.CombineFeatures(encodings, tokens[i-3:i])
		outputs := network.Prediction(inputs)
		maxIndex := 0
		maxValue := float32(0.0)
		for index, value := range outputs {
			if value > maxValue {
				maxValue = value
				maxIndex = index
			}
		}
		fmt.Print(labels[maxIndex], " ")
	}
}
