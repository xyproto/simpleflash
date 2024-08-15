package simpleflash

import (
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/xyproto/env/v2"
)

const (
	testModelName           = "gemini-1.5-flash-001" // "gemini-1.5-flash" might also work, as well as "gemini-1.0-pro-002" and "gemini-1.5-pro-001"
	testMultiModalModelName = "gemini-1.0-pro-vision-001"
	testLocation            = "europe-west4"
)

func TestMain(m *testing.M) {
	if env.No("PROJECT_ID") {
		fmt.Fprintln(os.Stderr, "PROJECT_ID environment variable is not set. Skipping tests.")
		os.Exit(0)
	}

	// Run the tests
	os.Exit(m.Run())
}

func TestNewSimpleFlash(t *testing.T) {
	sf, err := New(testModelName, testMultiModalModelName, testLocation, env.Str("PROJECT_ID"), false)
	if err != nil {
		t.Errorf("could not create a new SimpleFlash: %v", err)
	}

	if sf.ModelName != testModelName {
		t.Errorf("expected modelName to be '%s', got '%s'", testModelName, sf.ModelName)
	}

	if sf.MultiModalModelName != testMultiModalModelName {
		t.Errorf("expected multiModalModelName to be '%s', got '%s'", testMultiModalModelName, sf.MultiModalModelName)
	}

	if sf.ProjectLocation != testLocation {
		t.Errorf("expected projectLocation to be '%s', got '%s'", testLocation, sf.ProjectLocation)
	}

	if sf.ProjectID != env.Str("PROJECT_ID") {
		t.Errorf("expected projectID to be '%s', got '%s'", env.Str("PROJECT_ID"), sf.ProjectID)
	}
}

func TestSetTimeout(t *testing.T) {
	sf, err := New(testModelName, testMultiModalModelName, testLocation, env.Str("PROJECT_ID"), false)
	if err != nil {
		t.Errorf("could not create a new SimpleFlash: %v", err)
	}

	sf.Timeout = 10 * time.Second

	if sf.Timeout != 10*time.Second {
		t.Errorf("expected timeout to be 10s, got %v", sf.Timeout)
	}
}

func TestQueryGemini(t *testing.T) {
	sf, err := New(testModelName, testMultiModalModelName, testLocation, env.Str("PROJECT_ID"), true)
	if err != nil {
		t.Errorf("could not create a new SimpleFlash: %v", err)
	}

	sf.Timeout = 10 * time.Second

	// This is a placeholder test. In a real scenario, you would mock the VertexAI client and the response.
	result, err := sf.QueryGemini("Test prompt", nil, nil, nil)
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}

	if result == "" {
		t.Errorf("expected non-empty result, got empty string")
	}
}

func TestCountTextTokens(t *testing.T) {
	sf, err := New(testModelName, testMultiModalModelName, testLocation, env.Str("PROJECT_ID"), false)
	if err != nil {
		t.Errorf("could not create a new SimpleFlash: %v", err)
	}

	sf.Timeout = 10 * time.Second

	// This is a placeholder test. In a real scenario, you would mock the VertexAI client and the response.
	tokenCount, err := sf.CountTextTokens("Test prompt")
	if err != nil {
		t.Errorf("expected no error, got %v", err)
	}

	if tokenCount <= 0 {
		t.Errorf("expected token count to be greater than 0, got %d", tokenCount)
	}
}
