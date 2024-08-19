package main

import (
	"fmt"
	"os"
	"time"

	"github.com/xyproto/env/v2"
	"github.com/xyproto/simpleflash"
)

func main() {
	const (
		textModel       = "gemini-1.5-flash"
		//textModel       = "gemini-1.5-pro"
		multiModalModel = "gemini-1.0-pro-vision"
	)

	var (
		projectLocation = env.Str("PROJECT_LOCATION", "europe-west4") // europe-west4 is just the default
		projectID       = env.Str("PROJECT_ID")
	)

	if projectID == "" {
		fmt.Fprintln(os.Stderr, "Error: PROJECT_ID environment variable is not set.")
		return
	}

	sf, err := simpleflash.New(textModel, multiModalModel, projectLocation, projectID, true)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		return
	}

	sf.Timeout = 10 * time.Second

	const prompt = "Write a haiku about the color of cows."

	// Query Gemini with the prompt, nothing multimodal
	output, err := sf.QueryGemini(prompt, nil, nil, nil)
	if err != nil {
		fmt.Fprintln(os.Stderr, "Error:", err)
		return
	}

	fmt.Println(output)
}
