package adapter

import (
	"context"
	"encoding/json"
	"fmt"
	"iter"
	"log"
	"net/http"
	"time"

	"github.com/pkg/errors"
	openai "github.com/sashabaranov/go-openai"
	"google.golang.org/api/iterator"
	"google.golang.org/genai"

	"github.com/zhu327/gemini-openai-proxy/pkg/util"
)

const (
	GeminiPro       = "gemini-2.5-flash-lite-preview-06-17"
	GeminiProVision = "gemini-pro-vision"

	genaiRoleUser  = "user"
	genaiRoleModel = "model"
)

type GenaiModelAdapter interface {
	GenerateContent(ctx context.Context, req *ChatCompletionRequest) (*openai.ChatCompletionResponse, error)
	GenerateStreamContent(ctx context.Context, req *ChatCompletionRequest) (<-chan string, error)
}

type GeminiProAdapter struct {
	client *genai.Client
}

func NewGeminiProAdapter(client *genai.Client) GenaiModelAdapter {
	return &GeminiProAdapter{
		client: client,
	}
}

func (g *GeminiProAdapter) GenerateContent(
	ctx context.Context,
	req *ChatCompletionRequest,
) (*openai.ChatCompletionResponse, error) {
	cfg := &genai.GenerateContentConfig{}
	setGenaiModelByOpenaiRequest(cfg, req)

	var history []*genai.Content
	history = setChatHistory(history, req)

	cs, err := g.client.Chats.Create(ctx, GeminiPro, cfg, history)
	if err != nil {
		return nil, fmt.Errorf("gemini: chat: %q", err)
	}

	prompt := genai.Part{Text: req.Messages[len(req.Messages)-1].StringContent()}
	genaiResp, err := cs.SendMessage(ctx, prompt)
	if err != nil {
		return nil, errors.Wrap(err, "genai send message error")
	}

	openaiResp := genaiResponseToOpenaiResponse(genaiResp)
	return &openaiResp, nil
}

func (g *GeminiProAdapter) GenerateStreamContent(
	ctx context.Context,
	req *ChatCompletionRequest,
) (<-chan string, error) {
	cfg := &genai.GenerateContentConfig{}
	setGenaiModelByOpenaiRequest(cfg, req)
	var history []*genai.Content
	history = setChatHistory(history, req)

	cs, err := g.client.Chats.Create(ctx, GeminiPro, cfg, history)
	if err != nil {
		return nil, fmt.Errorf("gemini: create: %w", err)
	}

	prompt := genai.Part{
		Text: req.Messages[len(req.Messages)-1].StringContent(),
	}
	iter := cs.SendMessageStream(ctx, prompt)

	dataChan := make(chan string)
	go handleStreamIter(iter, dataChan)

	return dataChan, nil
}

type GeminiProVisionAdapter struct {
	client *genai.Client
}

func NewGeminiProVisionAdapter(client *genai.Client) GenaiModelAdapter {
	return &GeminiProVisionAdapter{
		client: client,
	}
}

func (g *GeminiProVisionAdapter) GenerateContent(
	ctx context.Context,
	req *ChatCompletionRequest,
) (*openai.ChatCompletionResponse, error) {
	cfg := &genai.GenerateContentConfig{}
	setGenaiModelByOpenaiRequest(cfg, req)

	// NOTE: use last message as prompt, gemini pro vision does not support context
	// https://ai.google.dev/tutorials/go_quickstart#multi-turn-conversations-chat
	prompt, err := g.openaiMessageToGenaiPrompt(req.Messages[len(req.Messages)-1])
	if err != nil {
		return nil, errors.Wrap(err, "genai generate prompt error")
	}

	genaiResp, err := g.client.Models.GenerateContent(
		ctx, GeminiProVision, []*genai.Content{
			prompt,
		}, cfg)
	if err != nil {
		return nil, errors.Wrap(err, "genai send message error")
	}

	openaiResp := genaiResponseToOpenaiResponse(genaiResp)
	return &openaiResp, nil
}

func (*GeminiProVisionAdapter) openaiMessageToGenaiPrompt(msg ChatCompletionMessage) (*genai.Content, error) {
	parts, err := msg.MultiContent()
	if err != nil {
		return nil, err
	}

	prompt := make([]*genai.Part, 0, len(parts))
	for _, part := range parts {
		switch part.Type {
		case openai.ChatMessagePartTypeText:
			prompt = append(prompt, &genai.Part{Text: part.Text})
		case openai.ChatMessagePartTypeImageURL:
			data, format, err := parseImageURL(part.ImageURL.URL)
			if err != nil {
				return nil, errors.Wrap(err, "parse image url error")
			}

			prompt = append(prompt, &genai.Part{
				InlineData: &genai.Blob{
					Data:     data,
					MIMEType: format,
				},
			})
		}
	}
	return &genai.Content{
		Parts: prompt,
	}, nil
}

func (g *GeminiProVisionAdapter) GenerateStreamContent(
	ctx context.Context,
	req *ChatCompletionRequest,
) (<-chan string, error) {
	cfg := &genai.GenerateContentConfig{}
	setGenaiModelByOpenaiRequest(cfg, req)

	// NOTE: use last message as prompt, gemini pro vision does not support context
	// https://ai.google.dev/tutorials/go_quickstart#multi-turn-conversations-chat
	prompt, err := g.openaiMessageToGenaiPrompt(req.Messages[len(req.Messages)-1])
	if err != nil {
		return nil, errors.Wrap(err, "genai generate prompt error")
	}

	iter := g.client.Models.GenerateContentStream(ctx, GeminiProVision, []*genai.Content{
		prompt,
	}, cfg)

	dataChan := make(chan string)
	go handleStreamIter(iter, dataChan)

	return dataChan, nil
}

func handleStreamIter(iter iter.Seq2[*genai.GenerateContentResponse, error], dataChan chan string) {
	defer close(dataChan)

	respID := util.GetUUID()
	created := time.Now().Unix()

	for genaiResp, err := range iter {
		if err != nil {
			if errors.Is(err, iterator.Done) {
				break
			}
			log.Printf("genai get stream message error %v\n", err)
			apiErr := openai.APIError{
				Code:    http.StatusInternalServerError,
				Message: err.Error(),
			}

			resp, _ := json.Marshal(apiErr)
			dataChan <- string(resp)
			break
		}

		openaiResp := genaiResponseToStreamCompletionResponse(genaiResp, respID, created)
		resp, _ := json.Marshal(openaiResp)
		dataChan <- string(resp)

		if len(openaiResp.Choices) > 0 && openaiResp.Choices[0].FinishReason != nil {
			break
		}
	}
}

func genaiResponseToStreamCompletionResponse(
	genaiResp *genai.GenerateContentResponse,
	respID string,
	created int64,
) *CompletionResponse {
	resp := CompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%s", respID),
		Object:  "chat.completion.chunk",
		Created: created,
		Model:   GeminiPro,
		Choices: make([]CompletionChoice, 0, len(genaiResp.Candidates)),
	}

	for i, candidate := range genaiResp.Candidates {
		var content string
		if candidate.Content != nil && len(candidate.Content.Parts) > 0 {
			content = candidate.Content.Parts[0].Text
		}

		choice := CompletionChoice{
			Index: i,
		}
		choice.Delta.Content = content

		if candidate.FinishReason > genai.FinishReasonStop {
			log.Printf("genai message finish reason %s\n", candidate.FinishReason)

			var openaiFinishReason string = string(openai.FinishReasonStop)
			if candidate.FinishReason == genai.FinishReasonMaxTokens {
				openaiFinishReason = string(openai.FinishReasonLength)
			}
			choice.FinishReason = &openaiFinishReason
		}

		resp.Choices = append(resp.Choices, choice)
	}
	return &resp
}

func genaiResponseToOpenaiResponse(
	genaiResp *genai.GenerateContentResponse,
) openai.ChatCompletionResponse {
	resp := openai.ChatCompletionResponse{
		ID:      fmt.Sprintf("chatcmpl-%s", util.GetUUID()),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   GeminiPro,
		Choices: make([]openai.ChatCompletionChoice, 0, len(genaiResp.Candidates)),
	}

	for i, candidate := range genaiResp.Candidates {
		var content string
		if candidate.Content != nil && len(candidate.Content.Parts) > 0 {
			content = candidate.Content.Parts[0].Text
		}

		choice := openai.ChatCompletionChoice{
			Index: i,
			Message: openai.ChatCompletionMessage{
				Role:    openai.ChatMessageRoleAssistant,
				Content: content,
			},
			FinishReason: openai.FinishReasonStop,
		}
		resp.Choices = append(resp.Choices, choice)
	}
	return resp
}

func setChatHistory(history []*genai.Content, req *ChatCompletionRequest) []*genai.Content {
	if len(req.Messages) > 1 {
		for _, message := range req.Messages[:len(req.Messages)-1] {
			switch message.Role {
			case openai.ChatMessageRoleSystem:
				history = append(history, []*genai.Content{
					{
						Parts: []*genai.Part{
							{Text: message.StringContent()},
						},
						Role: genaiRoleUser,
					},
					{
						Parts: []*genai.Part{
							{Text: "ok."},
						},
						Role: genaiRoleModel,
					},
				}...)
			case openai.ChatMessageRoleAssistant:
				history = append(history, &genai.Content{
					Parts: []*genai.Part{
						{Text: message.StringContent()},
					},
					Role: genaiRoleModel,
				})
			case openai.ChatMessageRoleUser:
				history = append(history, &genai.Content{
					Parts: []*genai.Part{
						{Text: message.StringContent()},
					},
					Role: genaiRoleUser,
				})
			}
		}
	}

	if len(history) != 0 && history[len(history)-1].Role != genaiRoleModel {
		history = append(history, &genai.Content{
			Parts: []*genai.Part{
				{Text: "ok."},
			},
			Role: genaiRoleModel,
		})
	}

	return history
}

func setGenaiModelByOpenaiRequest(cfg *genai.GenerateContentConfig, req *ChatCompletionRequest) {
	if req.MaxTokens != 0 {
		cfg.MaxOutputTokens = req.MaxTokens
	}
	if req.Temperature != 0 {
		cfg.Temperature = &req.Temperature
	}
	if req.TopP != 0 {
		cfg.TopP = &req.TopP
	}
	cfg.SafetySettings = []*genai.SafetySetting{
		{
			Category:  genai.HarmCategoryHarassment,
			Threshold: genai.HarmBlockThresholdBlockNone,
		},
		{
			Category:  genai.HarmCategoryHateSpeech,
			Threshold: genai.HarmBlockThresholdBlockNone,
		},
		{
			Category:  genai.HarmCategorySexuallyExplicit,
			Threshold: genai.HarmBlockThresholdBlockNone,
		},
		{
			Category:  genai.HarmCategoryDangerousContent,
			Threshold: genai.HarmBlockThresholdBlockNone,
		},
	}
}
