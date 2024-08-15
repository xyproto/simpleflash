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

type SimpleFlash struct {
	modelName           string
	multiModalModelName string
	projectLocation     string
	projectID           string
	client              *genai.Client
	cache               *bigcache.BigCache
	timeout             time.Duration
}

func New(modelName, multiModalModelName, projectLocation, projectID string, cache bool) *SimpleFlash {
	sf := &SimpleFlash{
		modelName:           env.Str("MODEL_NAME", modelName),
		multiModalModelName: env.Str("MULTI_MODAL_MODEL_NAME", multiModalModelName),
		projectLocation:     env.Str("PROJECT_LOCATION", projectLocation),
		projectID:           env.Str("PROJECT_ID", projectID),
		timeout:             3 * time.Minute,
	}

	// Initialize the genai client
	ctx := context.Background()
	creds, err := google.FindDefaultCredentials(ctx, "https://www.googleapis.com/auth/cloud-platform")
	if err != nil {
		log.Fatalf("Failed to obtain default credentials: %v", err)
	}
	genaiClient, err := genai.NewClient(ctx, sf.projectID, sf.projectLocation, option.WithCredentials(creds))
	if err != nil {
		log.Fatalf("failed to create genai client: %v", err)
	}
	sf.client = genaiClient

	// Initialize cache if the cache parameter is true
	if cache {
		err := sf.initCache()
		if err != nil {
			log.Fatalf("Failed to initialize cache: %v", err)
		}
	}

	return sf
}

// SetTimeout sets the timeout for requests
func (sf *SimpleFlash) SetTimeout(t time.Duration) {
	sf.timeout = t
}

// initCache initializes the BigCache cache
func (sf *SimpleFlash) initCache() error {
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
	sf.cache = c
	return nil
}

// QueryGemini processes a prompt with optional temperature, base64-encoded data, and MIME type for the data.
func (sf *SimpleFlash) QueryGemini(prompt string, temperature *float64, base64Data, dataMimeType *string) (string, error) {
	modelName := sf.modelName

	// Generate a unique cache key based on prompt and optionally on temperature, base64Data, and dataMimeType
	cacheKeyComponents := prompt
	if temperature != nil {
		cacheKeyComponents += fmt.Sprintf("%f", *temperature)
	}
	if base64Data != nil {
		cacheKeyComponents += *base64Data
		modelName = sf.multiModalModelName
	}
	if dataMimeType != nil {
		cacheKeyComponents += *dataMimeType
	}
	cacheKey := fmt.Sprintf("%x", sha256.Sum256([]byte(cacheKeyComponents)))

	// Check cache for existing entry
	if sf.cache != nil {
		if entry, err := sf.cache.Get(cacheKey); err == nil {
			return string(entry), nil
		}
	}

	// Initialize the multimodal instance, using the provided temperature if available
	mm := multimodal.New(modelName, 0.0) // Default temperature
	mm.SetTimeout(sf.timeout)
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

	ctx, cancel := context.WithTimeout(context.Background(), sf.timeout)
	defer cancel()

	// Submit the multimodal query and process the result
	result, err := mm.SubmitToClient(ctx, sf.client)
	if err != nil {
		return "", fmt.Errorf("failed to process response: %v", err)
	}

	result = strings.TrimSpace(result)

	// Store the new result in the cache
	if sf.cache != nil {
		_ = sf.cache.Set(cacheKey, []byte(result))
	}

	return result, nil
}

// CountTextTokens tries to count the number of tokens in the given prompt, using the VertexAI API
func (sf *SimpleFlash) CountTextTokens(prompt string) (int, error) {
	modelName := sf.modelName
	mm := multimodal.New(modelName, 0.0)
	mm.SetTimeout(sf.timeout)

	ctx, cancel := context.WithTimeout(context.Background(), sf.timeout)
	defer cancel()

	return mm.CountTextTokensWithClient(ctx, sf.client, prompt)
}
