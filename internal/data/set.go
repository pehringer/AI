package data

type (
	Sample struct {
		Features []float32
		Targets  []float32
	}
	Set        []Sample
	Tokens     []int
	Vocabulary map[string]int
)
