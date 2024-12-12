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

	trainingSamples, vocabulary := data.Cbow(reader, 3)
	fmt.Println(len(trainingSamples), len(vocabulary))
	network := snn.Initialize(len(vocabulary), 24, len(vocabulary))
	network.BatchTrain(trainingSamples, 300, 12, 0.05)

	file, _ = os.Open("text/TheForgottenVillage.txt")
	reader = bufio.NewReader(file)
	tokens, _ := data.TokenizeText(reader)
	labels := data.GetLabels(vocabulary)
	encodings := data.HotEncodings(len(vocabulary))
	for i := 3; i < len(tokens)-3; i++ {
		context := append([]int{}, tokens[i-3:i]...)
		context = append(context, tokens[i+1:i+4]...)
		inputs := data.CombineFeatures(encodings, context)
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
