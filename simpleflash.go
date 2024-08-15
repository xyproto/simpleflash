package simpleflash

import (
	"context"
	"crypto/sha256"
	"encoding/base64"
	"fmt"
	"log"
	"strings"
	"time"

	"cloud.google.com/go/vertexai/genai"
	"github.com/allegro/bigcache/v3"
	"github.com/xyproto/env"
	"github.com/xyproto/multimodal"
	"golang.org/x/oauth2/google"
	"google.golang.org/api/option"
)

func init() {
	go func() {
		InitTranslationCache()
	}()
}

const (
	temperature         = 0.0
	textModelName       = "gemini-1.5-flash-001"
	multiModalModelName = "gemini-1.0-pro-vision" // TODO: update
)

var (
	projectLocation = env.Str("PROJECT_LOCATION", "europe-west4")
	projectID       = env.Str("PROJECT_ID", "44444444444")

	timeout = 3 * time.Minute
	verbose = env.Bool("VERBOSE")

	queryCache *bigcache.BigCache = nil

	// Create credentials using Google's default credentials
	genaiClient = func() *genai.Client {
		ctx := context.Background()
		creds, err := google.FindDefaultCredentials(ctx, "https://www.googleapis.com/auth/cloud-platform")
		if err != nil {
			log.Fatalf("Failed to obtain default credentials: %v", err)
		}
		// Create a genai.Client using the credentials
		genaiClient, err := genai.NewClient(ctx, projectID, projectLocation, option.WithCredentials(creds))
		if err != nil {
			log.Fatalf("failed to create genai client: %v", err)
		}
		return genaiClient
	}()
)

func SetTimeout(timeout time.Duration) {
	timeout = timeout
}

// InitTranslationCache initializes the BigCache cache
func InitTranslationCache() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	config := bigcache.DefaultConfig(24 * time.Hour)
	config.HardMaxCacheSize = 256 // MB
	config.StatsEnabled = false
	config.Verbose = false
	c, err := bigcache.New(ctx, config)
	if err != nil {
		return err
	}
	queryCache = c
	return nil
}

// QueryGemini processes a prompt with optional temperature, base64-encoded data, and MIME type for the data.
// The temperature, base64data, and dataMimeType are optional. The function is generic to handle any type of data.
func QueryGemini(prompt string, temperature *float64, base64Data, dataMimeType, customModelName *string) (string, error) {
	modelName := textModelName
	if customModelName != nil {
		modelName = *customModelName
	}

	// Generate a unique cache key based on prompt and optionally on temperature, base64Data, and dataMimeType
	cacheKeyComponents := prompt
	if temperature != nil {
		cacheKeyComponents += fmt.Sprintf("%f", *temperature)
	}
	if base64Data != nil {
		cacheKeyComponents += *base64Data
		modelName = multiModalModelName
		if customModelName != nil {
			modelName = *customModelName
		}
	}
	if dataMimeType != nil {
		cacheKeyComponents += *dataMimeType
	}
	if customModelName != nil {
		cacheKeyComponents += *customModelName
	}
	cacheKey := fmt.Sprintf("%x", sha256.Sum256([]byte(cacheKeyComponents)))

	// Check cache for existing entry
	if queryCache != nil {
		if entry, err := queryCache.Get(cacheKey); err == nil {
			return string(entry), nil
		}
	}

	// Initialize the multimodal instance, using the provided temperature if available
	mm := multimodal.New(modelName, 0.0) // Default temperature
	mm.SetTimeout(timeout)
	if temperature != nil {
		mm = multimodal.New(modelName, float32(*temperature))
	}
	mm.AddText(prompt)

	// If base64Data and dataMimeType are provided, decode the data and add it to the multimodal instance
	if base64Data != nil && dataMimeType != nil {
		data, err := base64.StdEncoding.DecodeString(*base64Data)
		if err != nil {
			return "", fmt.Errorf("failed to decode base64 data: %v", err)
		}
		mm.AddData(*dataMimeType, data)
	}

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	// Submit the multimodal query and process the result
	result, err := mm.SubmitToClient(ctx, genaiClient)
	if err != nil {
		return "", fmt.Errorf("failed to process response: %v", err)
	}

	result = strings.TrimSpace(result)

	// Store the new result in the cache
	if queryCache != nil {
		_ = queryCache.Set(cacheKey, []byte(result))
	}

	return result, nil
}

// CountTextTokens tries to count the number of tokens in the given prompt, using the VertexAI API
func CountTextTokens(prompt string, customModelName *string) (int, error) {
	modelName := textModelName
	if customModelName != nil {
		modelName = *customModelName
	}
	mm := multimodal.New(modelName, 0.0)
	mm.SetTimeout(timeout)

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	return mm.CountTextTokensWithClient(ctx, genaiClient, prompt)
}
