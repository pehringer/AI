package main

import (
	"bufio"
	"fmt"
	"strings"

	"github.com/pehringer/gobed/internal/data"
	"github.com/pehringer/gobed/internal/snn"
)

var (
	or = data.Set{
		{
			Features: []float32{0.0, 0.0},
			Targets:  []float32{0.0, 1.0},
		},
		{
			Features: []float32{0.0, 1.0},
			Targets:  []float32{1.0, 0.0},
		},
		{
			Features: []float32{1.0, 0.0},
			Targets:  []float32{1.0, 0.0},
		},
		{
			Features: []float32{1.0, 1.0},
			Targets:  []float32{1.0, 0.0},
		},
	}
	nor = data.Set{
		{
			Features: []float32{0.0, 0.0},
			Targets:  []float32{1.0, 0.0},
		},
		{
			Features: []float32{0.0, 1.0},
			Targets:  []float32{0.0, 1.0},
		},
		{
			Features: []float32{1.0, 0.0},
			Targets:  []float32{0.0, 1.0},
		},
		{
			Features: []float32{1.0, 1.0},
			Targets:  []float32{0.0, 1.0},
		},
	}
	xor = data.Set{
		{
			Features: []float32{0.0, 0.0},
			Targets:  []float32{0.0, 1.0},
		},
		{
			Features: []float32{0.0, 1.0},
			Targets:  []float32{1.0, 0.0},
		},
		{
			Features: []float32{1.0, 0.0},
			Targets:  []float32{1.0, 0.0},
		},
		{
			Features: []float32{1.0, 1.0},
			Targets:  []float32{0.0, 1.0},
		},
	}
	and = data.Set{
		{
			Features: []float32{0.0, 0.0},
			Targets:  []float32{0.0, 1.0},
		},
		{
			Features: []float32{0.0, 1.0},
			Targets:  []float32{0.0, 1.0},
		},
		{
			Features: []float32{1.0, 0.0},
			Targets:  []float32{0.0, 1.0},
		},
		{
			Features: []float32{1.0, 1.0},
			Targets:  []float32{1.0, 0.0},
		},
	}
	nand = data.Set{
		{
			Features: []float32{0.0, 0.0},
			Targets:  []float32{1.0, 0.0},
		},
		{
			Features: []float32{0.0, 1.0},
			Targets:  []float32{1.0, 0.0},
		},
		{
			Features: []float32{1.0, 0.0},
			Targets:  []float32{1.0, 0.0},
		},
		{
			Features: []float32{1.0, 1.0},
			Targets:  []float32{0.0, 1.0},
		},
	}
)

func main() {
	text := `The journey begins! 🚀 In 2024, humanity embarks on a new chapter: exploring Mars.
	With challenges (like radiation, supplies, and distance), the @NASA team, SpaceX, and other organizations aim high—very high.
	
	"To infinity & beyond!" - Buzz Lightyear's motto fits perfectly. But: is it worth it? 🤔
	Cost: $1,000,000,000+ per mission. Risks? High. Rewards? Potentially groundbreaking.
	
	Meanwhile, here on Earth 🌍, AI & robotics advance at lightning speed! Python > Java? Maybe.
	However, questions like: "Will AI replace humans?" or "Should we fear AI?" persist. 😅
	
	Fun facts:
	1. The average person blinks ~20,000 times/day.
	2. Honey never spoils. (Yes, NEVER!)
	3. A shrimp's heart is in its head. 🤯
	
	**Stay curious.** Learn, explore, grow... and maybe, one day, you’ll touch the stars. ✨`
	reader := bufio.NewReader(strings.NewReader(text))
	lookup, tokens := data.TokenizeText(reader)
	lookdown := map[int]string{}
	for key, value := range lookup {
		lookdown[value] = key
	}
	for _, token := range tokens {
		fmt.Print(lookdown[token], " ")
	}
	fmt.Println()

	ts := nand
	n := snn.Initialize(2, 4, 2)
	//n.OnlineTrain(ts, 4096, 0.05)
	n.BatchTrain(ts, 25000, 3, 0.05)
	for i := 0; i < len(ts); i++ {
		fmt.Println(ts[i].Features, ts[i].Targets, n.Prediction(ts[i].Features))
	}
}
