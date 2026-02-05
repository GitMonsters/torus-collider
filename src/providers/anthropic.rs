//! Anthropic Claude API provider.
//!
//! This provider connects to the Anthropic API for Claude models.

use super::traits::LLMProvider;
use super::types::{
    ChatCompletion, CompletionOptions, FinishReason, Message, ProviderError, ProviderResult, Role,
    ToolCall, ToolDefinition, Usage,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

/// Anthropic Claude API provider.
///
/// # Example
///
/// ```rust,ignore
/// use torus_attention::providers::{AnthropicProvider, Message, CompletionOptions};
///
/// let provider = AnthropicProvider::new("your-api-key", "claude-3-opus-20240229")?;
/// let messages = vec![Message::user("Hello!")];
/// let response = provider.complete(&messages, CompletionOptions::default()).await?;
/// ```
pub struct AnthropicProvider {
    /// API key
    api_key: String,
    /// Model ID
    model: String,
    /// Base URL
    base_url: String,
    /// HTTP client
    client: reqwest::Client,
    /// API version
    api_version: String,
}

impl AnthropicProvider {
    /// Create a new Anthropic provider
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> ProviderResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .map_err(|e| ProviderError::Internal(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            api_key: api_key.into(),
            model: model.into(),
            base_url: "https://api.anthropic.com".to_string(),
            client,
            api_version: "2023-06-01".to_string(),
        })
    }

    /// Create with custom base URL (for proxies or testing)
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Create with custom API version
    pub fn with_api_version(mut self, version: impl Into<String>) -> Self {
        self.api_version = version.into();
        self
    }

    /// Build the API request
    fn build_request(
        &self,
        messages: &[Message],
        options: &CompletionOptions,
    ) -> ProviderResult<AnthropicRequest> {
        // Extract system message if present
        let system = messages
            .iter()
            .find(|m| m.role == Role::System)
            .map(|m| m.content.clone());

        // Convert messages (excluding system)
        let api_messages: Vec<AnthropicMessage> = messages
            .iter()
            .filter(|m| m.role != Role::System)
            .map(|m| AnthropicMessage {
                role: match m.role {
                    Role::User => "user".to_string(),
                    Role::Assistant => "assistant".to_string(),
                    Role::Tool => "user".to_string(), // Tool results go as user messages
                    Role::System => unreachable!(),
                },
                content: m.content.clone(),
            })
            .collect();

        // Convert tools
        let tools: Option<Vec<AnthropicTool>> = if options.tools.is_empty() {
            None
        } else {
            Some(
                options
                    .tools
                    .iter()
                    .map(|t| AnthropicTool {
                        name: t.name.clone(),
                        description: t.description.clone(),
                        input_schema: t.parameters.clone(),
                    })
                    .collect(),
            )
        };

        Ok(AnthropicRequest {
            model: self.model.clone(),
            messages: api_messages,
            system,
            max_tokens: options.max_tokens,
            temperature: Some(options.temperature),
            top_p: Some(options.top_p),
            stop_sequences: if options.stop.is_empty() {
                None
            } else {
                Some(options.stop.clone())
            },
            stream: Some(options.stream),
            tools,
        })
    }

    /// Parse the API response
    fn parse_response(&self, response: AnthropicResponse) -> ProviderResult<ChatCompletion> {
        // Extract text content
        let mut content = String::new();
        let mut tool_calls = Vec::new();

        for block in &response.content {
            match block {
                ContentBlock::Text { text } => {
                    content.push_str(text);
                }
                ContentBlock::ToolUse { id, name, input } => {
                    tool_calls.push(ToolCall {
                        id: id.clone(),
                        name: name.clone(),
                        arguments: input.to_string(),
                    });
                }
            }
        }

        let finish_reason = match response.stop_reason.as_deref() {
            Some("end_turn") => FinishReason::Stop,
            Some("max_tokens") => FinishReason::Length,
            Some("stop_sequence") => FinishReason::StopSequence,
            Some("tool_use") => FinishReason::ToolCalls,
            _ => FinishReason::Other,
        };

        let mut message = Message::assistant(content);
        if !tool_calls.is_empty() {
            message.tool_calls = Some(tool_calls);
        }

        Ok(ChatCompletion {
            id: response.id,
            model: response.model,
            message,
            finish_reason,
            usage: Usage {
                prompt_tokens: response.usage.input_tokens,
                completion_tokens: response.usage.output_tokens,
                total_tokens: response.usage.input_tokens + response.usage.output_tokens,
                cached_tokens: response.usage.cache_read_input_tokens,
            },
            metadata: HashMap::new(),
        })
    }
}

impl LLMProvider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    fn model(&self) -> &str {
        &self.model
    }

    fn is_available(&self) -> bool {
        !self.api_key.is_empty()
    }

    fn complete<'a>(
        &'a self,
        messages: &'a [Message],
        options: CompletionOptions,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<ChatCompletion>> + Send + 'a>> {
        Box::pin(async move {
            let request = self.build_request(messages, &options)?;

            let response = self
                .client
                .post(format!("{}/v1/messages", self.base_url))
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", &self.api_version)
                .header("content-type", "application/json")
                .json(&request)
                .send()
                .await
                .map_err(|e| {
                    if e.is_timeout() {
                        ProviderError::Timeout(e.to_string())
                    } else if e.is_connect() {
                        ProviderError::NetworkError(e.to_string())
                    } else {
                        ProviderError::NetworkError(e.to_string())
                    }
                })?;

            let status = response.status();

            if !status.is_success() {
                let error_body = response.text().await.unwrap_or_default();

                return Err(match status.as_u16() {
                    401 => ProviderError::AuthenticationError(error_body),
                    429 => {
                        // Parse retry-after header if present
                        ProviderError::RateLimitError {
                            retry_after_secs: None,
                            message: error_body,
                        }
                    }
                    400 => ProviderError::InvalidRequest(error_body),
                    404 => ProviderError::ModelNotFound(self.model.clone()),
                    _ => ProviderError::ProviderSpecific {
                        code: status.to_string(),
                        message: error_body,
                    },
                });
            }

            let api_response: AnthropicResponse = response.json().await.map_err(|e| {
                ProviderError::ParseError(format!("Failed to parse response: {}", e))
            })?;

            self.parse_response(api_response)
        })
    }

    fn max_context_length(&self) -> usize {
        // Claude 3 models have 200k context
        if self.model.contains("claude-3") {
            200_000
        } else if self.model.contains("claude-2") {
            100_000
        } else {
            100_000
        }
    }
}

// =============================================================================
// API TYPES
// =============================================================================

#[derive(Debug, Serialize)]
struct AnthropicRequest {
    model: String,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    max_tokens: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop_sequences: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<AnthropicTool>>,
}

#[derive(Debug, Serialize)]
struct AnthropicMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct AnthropicResponse {
    id: String,
    model: String,
    content: Vec<ContentBlock>,
    stop_reason: Option<String>,
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: usize,
    output_tokens: usize,
    #[serde(default)]
    cache_read_input_tokens: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = AnthropicProvider::new("test-key", "claude-3-opus-20240229");
        assert!(provider.is_ok());
        let provider = provider.unwrap();
        assert_eq!(provider.name(), "anthropic");
        assert_eq!(provider.model(), "claude-3-opus-20240229");
    }

    #[test]
    fn test_is_available() {
        let provider = AnthropicProvider::new("test-key", "claude-3-opus-20240229").unwrap();
        assert!(provider.is_available());

        let empty_provider = AnthropicProvider::new("", "claude-3-opus-20240229").unwrap();
        assert!(!empty_provider.is_available());
    }

    #[test]
    fn test_max_context_length() {
        let claude3 = AnthropicProvider::new("key", "claude-3-opus-20240229").unwrap();
        assert_eq!(claude3.max_context_length(), 200_000);

        let claude2 = AnthropicProvider::new("key", "claude-2.1").unwrap();
        assert_eq!(claude2.max_context_length(), 100_000);
    }

    #[test]
    fn test_build_request() {
        let provider = AnthropicProvider::new("key", "claude-3-opus-20240229").unwrap();
        let messages = vec![
            Message::system("You are helpful"),
            Message::user("Hello!"),
        ];
        let options = CompletionOptions::default();

        let request = provider.build_request(&messages, &options).unwrap();
        assert_eq!(request.model, "claude-3-opus-20240229");
        assert_eq!(request.system, Some("You are helpful".to_string()));
        assert_eq!(request.messages.len(), 1); // Only user message, system is separate
    }
}
