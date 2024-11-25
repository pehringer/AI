package main

import (
	"fmt"

	"github.com/pehringer/gobed/internal/ssn"
	"github.com/pehringer/gobed/internal/vec"
)

func main() {
	a := []float32{1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9}
	b := []float32{0.0, 1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8}
	var c float32
	vec.Dot(a, b, &c)
	fmt.Println(a)
	fmt.Println(b)
	fmt.Println(c)
	ssn.New(10000, 300, 10000)
}
