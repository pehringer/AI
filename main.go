package main

import (
	"bufio"
	"fmt"
	"os"

	"github.com/pehringer/gobed/internal/data"
	"github.com/pehringer/gobed/internal/snn"
	"github.com/pehringer/gobed/internal/test"
)

func main() {
	va := []float32{1, 2, 3}
	vb := []float32{4, 5, 6}
	cs := test.CosineSimilarity(va, vb)
	fmt.Println("cos sim:", cs)

	file, _ := os.Open("text/TheForgottenVillage.txt")
	reader := bufio.NewReader(file)

	trainingSamples, vocabulary := data.Cbow(reader, 3)
	network := snn.Initialize(len(vocabulary), 24, len(vocabulary))
	network.BatchTrain(trainingSamples, 1, 12, 0.05)
}
