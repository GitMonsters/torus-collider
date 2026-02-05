//! OpenAI GPT API provider.
//!
//! This provider connects to the OpenAI API for GPT models.

use super::traits::LLMProvider;
use super::types::{
    ChatCompletion, CompletionOptions, FinishReason, Message, ProviderError, ProviderResult, Role,
    ToolCall, Usage,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;

/// OpenAI GPT API provider.
///
/// # Example
///
/// ```rust,ignore
/// use torus_attention::providers::{OpenAIProvider, Message, CompletionOptions};
///
/// let provider = OpenAIProvider::new("your-api-key", "gpt-4")?;
/// let messages = vec![Message::user("Hello!")];
/// let response = provider.complete(&messages, CompletionOptions::default()).await?;
/// ```
pub struct OpenAIProvider {
    /// API key
    api_key: String,
    /// Model ID
    model: String,
    /// Base URL
    base_url: String,
    /// HTTP client
    client: reqwest::Client,
    /// Organization ID (optional)
    organization: Option<String>,
}

impl OpenAIProvider {
    /// Create a new OpenAI provider
    pub fn new(api_key: impl Into<String>, model: impl Into<String>) -> ProviderResult<Self> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(120))
            .build()
            .map_err(|e| ProviderError::Internal(format!("Failed to create HTTP client: {}", e)))?;

        Ok(Self {
            api_key: api_key.into(),
            model: model.into(),
            base_url: "https://api.openai.com".to_string(),
            client,
            organization: None,
        })
    }

    /// Create with custom base URL (for Azure, proxies, or testing)
    pub fn with_base_url(mut self, base_url: impl Into<String>) -> Self {
        self.base_url = base_url.into();
        self
    }

    /// Set organization ID
    pub fn with_organization(mut self, org: impl Into<String>) -> Self {
        self.organization = Some(org.into());
        self
    }

    /// Build the API request
    fn build_request(
        &self,
        messages: &[Message],
        options: &CompletionOptions,
    ) -> ProviderResult<OpenAIRequest> {
        // Convert messages
        let api_messages: Vec<OpenAIMessage> = messages
            .iter()
            .map(|m| OpenAIMessage {
                role: match m.role {
                    Role::System => "system".to_string(),
                    Role::User => "user".to_string(),
                    Role::Assistant => "assistant".to_string(),
                    Role::Tool => "tool".to_string(),
                },
                content: Some(m.content.clone()),
                name: m.name.clone(),
                tool_call_id: m.tool_call_id.clone(),
                tool_calls: None, // Set separately for assistant messages with tool calls
            })
            .collect();

        // Convert tools
        let tools: Option<Vec<OpenAITool>> = if options.tools.is_empty() {
            None
        } else {
            Some(
                options
                    .tools
                    .iter()
                    .map(|t| OpenAITool {
                        tool_type: "function".to_string(),
                        function: OpenAIFunction {
                            name: t.name.clone(),
                            description: t.description.clone(),
                            parameters: t.parameters.clone(),
                        },
                    })
                    .collect(),
            )
        };

        Ok(OpenAIRequest {
            model: self.model.clone(),
            messages: api_messages,
            max_tokens: Some(options.max_tokens),
            temperature: Some(options.temperature),
            top_p: Some(options.top_p),
            stop: if options.stop.is_empty() {
                None
            } else {
                Some(options.stop.clone())
            },
            presence_penalty: Some(options.presence_penalty),
            frequency_penalty: Some(options.frequency_penalty),
            stream: Some(options.stream),
            tools,
            user: options.user.clone(),
        })
    }

    /// Parse the API response
    fn parse_response(&self, response: OpenAIResponse) -> ProviderResult<ChatCompletion> {
        let choice = response
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| ProviderError::ParseError("No choices in response".to_string()))?;

        let finish_reason = match choice.finish_reason.as_deref() {
            Some("stop") => FinishReason::Stop,
            Some("length") => FinishReason::Length,
            Some("tool_calls") => FinishReason::ToolCalls,
            Some("content_filter") => FinishReason::ContentFilter,
            _ => FinishReason::Other,
        };

        let mut message = Message::assistant(choice.message.content.unwrap_or_default());

        // Handle tool calls
        if let Some(tool_calls) = choice.message.tool_calls {
            message.tool_calls = Some(
                tool_calls
                    .into_iter()
                    .map(|tc| ToolCall {
                        id: tc.id,
                        name: tc.function.name,
                        arguments: tc.function.arguments,
                    })
                    .collect(),
            );
        }

        Ok(ChatCompletion {
            id: response.id,
            model: response.model,
            message,
            finish_reason,
            usage: Usage {
                prompt_tokens: response.usage.prompt_tokens,
                completion_tokens: response.usage.completion_tokens,
                total_tokens: response.usage.total_tokens,
                cached_tokens: None,
            },
            metadata: HashMap::new(),
        })
    }
}

impl LLMProvider for OpenAIProvider {
    fn name(&self) -> &str {
        "openai"
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

            let mut req = self
                .client
                .post(format!("{}/v1/chat/completions", self.base_url))
                .header("Authorization", format!("Bearer {}", self.api_key))
                .header("Content-Type", "application/json");

            if let Some(ref org) = self.organization {
                req = req.header("OpenAI-Organization", org);
            }

            let response = req.json(&request).send().await.map_err(|e| {
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
                    429 => ProviderError::RateLimitError {
                        retry_after_secs: None,
                        message: error_body,
                    },
                    400 => ProviderError::InvalidRequest(error_body),
                    404 => ProviderError::ModelNotFound(self.model.clone()),
                    _ => ProviderError::ProviderSpecific {
                        code: status.to_string(),
                        message: error_body,
                    },
                });
            }

            let api_response: OpenAIResponse = response.json().await.map_err(|e| {
                ProviderError::ParseError(format!("Failed to parse response: {}", e))
            })?;

            self.parse_response(api_response)
        })
    }

    fn max_context_length(&self) -> usize {
        // Context lengths for various models
        if self.model.contains("gpt-4-turbo") || self.model.contains("gpt-4o") {
            128_000
        } else if self.model.contains("gpt-4-32k") {
            32_768
        } else if self.model.contains("gpt-4") {
            8_192
        } else if self.model.contains("gpt-3.5-turbo-16k") {
            16_384
        } else if self.model.contains("gpt-3.5") {
            4_096
        } else {
            4_096
        }
    }
}

// =============================================================================
// API TYPES
// =============================================================================

#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    presence_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    frequency_penalty: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<OpenAITool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    user: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
}

#[derive(Debug, Serialize)]
struct OpenAITool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAIFunction,
}

#[derive(Debug, Serialize)]
struct OpenAIFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    id: String,
    model: String,
    choices: Vec<OpenAIChoice>,
    usage: OpenAIUsage,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIToolCall {
    id: String,
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAIToolCallFunction,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIToolCallFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: usize,
    completion_tokens: usize,
    total_tokens: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_provider_creation() {
        let provider = OpenAIProvider::new("test-key", "gpt-4");
        assert!(provider.is_ok());
        let provider = provider.unwrap();
        assert_eq!(provider.name(), "openai");
        assert_eq!(provider.model(), "gpt-4");
    }

    #[test]
    fn test_is_available() {
        let provider = OpenAIProvider::new("test-key", "gpt-4").unwrap();
        assert!(provider.is_available());

        let empty_provider = OpenAIProvider::new("", "gpt-4").unwrap();
        assert!(!empty_provider.is_available());
    }

    #[test]
    fn test_max_context_length() {
        let gpt4 = OpenAIProvider::new("key", "gpt-4").unwrap();
        assert_eq!(gpt4.max_context_length(), 8_192);

        let gpt4_turbo = OpenAIProvider::new("key", "gpt-4-turbo").unwrap();
        assert_eq!(gpt4_turbo.max_context_length(), 128_000);

        let gpt35 = OpenAIProvider::new("key", "gpt-3.5-turbo").unwrap();
        assert_eq!(gpt35.max_context_length(), 4_096);
    }

    #[test]
    fn test_build_request() {
        let provider = OpenAIProvider::new("key", "gpt-4").unwrap();
        let messages = vec![
            Message::system("You are helpful"),
            Message::user("Hello!"),
        ];
        let options = CompletionOptions::default();

        let request = provider.build_request(&messages, &options).unwrap();
        assert_eq!(request.model, "gpt-4");
        assert_eq!(request.messages.len(), 2);
        assert_eq!(request.messages[0].role, "system");
        assert_eq!(request.messages[1].role, "user");
    }

    #[test]
    fn test_with_organization() {
        let provider = OpenAIProvider::new("key", "gpt-4")
            .unwrap()
            .with_organization("org-123");
        assert_eq!(provider.organization, Some("org-123".to_string()));
    }
}
