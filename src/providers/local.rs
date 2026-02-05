//! Local provider using TorusLLM.
//!
//! This provider wraps the local TorusLLM model for inference.

use super::traits::LLMProvider;
use super::types::{
    ChatCompletion, CompletionOptions, FinishReason, Message, ProviderError, ProviderResult, Role,
    Usage,
};
use crate::llm::{TorusLLM, TorusLLMConfig};
use crate::TorusResult;
use candle_core::Device;
use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

/// Local provider using TorusLLM for inference.
///
/// This provider runs the model locally on the device (CPU/GPU).
///
/// # Example
///
/// ```rust,ignore
/// use torus_attention::providers::{LocalProvider, Message, CompletionOptions};
///
/// let provider = LocalProvider::new_random(TorusLLMConfig::tiny())?;
/// let messages = vec![Message::user("Hello!")];
/// let response = provider.complete(&messages, CompletionOptions::default()).await?;
/// ```
pub struct LocalProvider {
    /// The underlying model
    model: Arc<RwLock<TorusLLM>>,
    /// Configuration
    config: TorusLLMConfig,
    /// Model name
    name: String,
    /// Device
    device: Device,
}

impl LocalProvider {
    /// Create a new local provider with an existing model
    pub fn new(model: TorusLLM, name: impl Into<String>) -> Self {
        let config = model.config().clone();
        let device = model.device().clone();
        Self {
            model: Arc::new(RwLock::new(model)),
            config,
            name: name.into(),
            device,
        }
    }

    /// Create a new local provider with random weights (for testing)
    pub fn new_random(config: TorusLLMConfig) -> TorusResult<Self> {
        let device = Device::Cpu;
        let (model, _varmap) = TorusLLM::new_random(config.clone(), &device)?;
        Ok(Self {
            model: Arc::new(RwLock::new(model)),
            config,
            name: "torus-local".to_string(),
            device,
        })
    }

    /// Create from a checkpoint
    pub fn from_checkpoint(
        config: TorusLLMConfig,
        checkpoint_path: &str,
        device: &Device,
    ) -> TorusResult<Self> {
        use candle_nn::VarBuilder;
        use std::path::Path;

        let path = Path::new(checkpoint_path);
        if !path.exists() {
            return Err(crate::error::TorusError::InvalidParameter(format!(
                "Checkpoint not found: {}",
                checkpoint_path
            )));
        }

        // Load the safetensors file
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], candle_core::DType::F32, device)? };
        let model = TorusLLM::new(config.clone(), vb)?;

        Ok(Self {
            model: Arc::new(RwLock::new(model)),
            config,
            name: format!("torus-{}", path.file_stem().unwrap_or_default().to_string_lossy()),
            device: device.clone(),
        })
    }

    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Get the configuration
    pub fn config(&self) -> &TorusLLMConfig {
        &self.config
    }

    /// Format messages into a prompt string
    fn format_prompt(&self, messages: &[Message]) -> String {
        // Simple chat format - could be customized per model
        let mut prompt = String::new();

        for msg in messages {
            match msg.role {
                Role::System => {
                    prompt.push_str("<|system|>\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("\n<|end|>\n");
                }
                Role::User => {
                    prompt.push_str("<|user|>\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("\n<|end|>\n");
                }
                Role::Assistant => {
                    prompt.push_str("<|assistant|>\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("\n<|end|>\n");
                }
                Role::Tool => {
                    prompt.push_str("<|tool|>\n");
                    prompt.push_str(&msg.content);
                    prompt.push_str("\n<|end|>\n");
                }
            }
        }

        // Add prompt for assistant response
        prompt.push_str("<|assistant|>\n");
        prompt
    }

    /// Simple tokenization (character-level for now)
    /// In production, use a proper tokenizer
    fn tokenize(&self, text: &str) -> Vec<u32> {
        text.chars()
            .map(|c| (c as u32) % self.config.vocab_size as u32)
            .collect()
    }

    /// Detokenize (reverse of tokenize)
    fn detokenize(&self, tokens: &[u32]) -> String {
        tokens
            .iter()
            .filter_map(|&t| char::from_u32(t))
            .collect()
    }
}

impl LLMProvider for LocalProvider {
    fn name(&self) -> &str {
        &self.name
    }

    fn model(&self) -> &str {
        "torus-llm"
    }

    fn is_available(&self) -> bool {
        true // Local model is always available
    }

    fn complete<'a>(
        &'a self,
        messages: &'a [Message],
        options: CompletionOptions,
    ) -> Pin<Box<dyn Future<Output = ProviderResult<ChatCompletion>> + Send + 'a>> {
        Box::pin(async move {
            // Format prompt
            let prompt = self.format_prompt(messages);
            let input_tokens = self.tokenize(&prompt);
            let prompt_token_count = input_tokens.len();

            // Check token limit
            if prompt_token_count > self.config.max_seq_len {
                return Err(ProviderError::TokenLimitExceeded {
                    limit: self.config.max_seq_len,
                    requested: prompt_token_count,
                });
            }

            // Generate
            let generated = {
                let model = self.model.read().map_err(|e| {
                    ProviderError::Internal(format!("Failed to acquire model lock: {}", e))
                })?;

                // Convert to tensor
                let input_ids = candle_core::Tensor::new(input_tokens.as_slice(), &self.device)
                    .map_err(|e| ProviderError::Internal(format!("Tensor error: {}", e)))?
                    .unsqueeze(0)
                    .map_err(|e| ProviderError::Internal(format!("Unsqueeze error: {}", e)))?;

                // For now, just do a forward pass and get top token
                // In production, implement proper generation loop
                let _logits = model.forward(&input_ids).map_err(|e| {
                    ProviderError::Internal(format!("Forward pass error: {}", e))
                })?;

                // Placeholder: return a simple response
                // Real implementation would sample from logits
                format!("I received your message with {} tokens. This is a placeholder response from the local TorusLLM model.", prompt_token_count)
            };

            let completion_tokens = generated.len() / 4; // Rough estimate

            Ok(ChatCompletion {
                id: format!("local-{}", uuid_simple()),
                model: self.name.clone(),
                message: Message::assistant(generated),
                finish_reason: FinishReason::Stop,
                usage: Usage::new(prompt_token_count, completion_tokens),
                metadata: HashMap::new(),
            })
        })
    }

    fn count_tokens(&self, messages: &[Message]) -> ProviderResult<usize> {
        let prompt = self.format_prompt(messages);
        Ok(self.tokenize(&prompt).len())
    }

    fn max_context_length(&self) -> usize {
        self.config.max_seq_len
    }
}

/// Simple UUID generator (not cryptographically secure)
fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    format!("{:x}{:x}", now.as_secs(), now.subsec_nanos())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_local_provider_creation() {
        let provider = LocalProvider::new_random(TorusLLMConfig::tiny());
        assert!(provider.is_ok());
        let provider = provider.unwrap();
        assert!(provider.is_available());
        assert_eq!(provider.model(), "torus-llm");
    }

    #[test]
    fn test_format_prompt() {
        let provider = LocalProvider::new_random(TorusLLMConfig::tiny()).unwrap();
        let messages = vec![
            Message::system("You are helpful"),
            Message::user("Hello!"),
        ];
        let prompt = provider.format_prompt(&messages);
        assert!(prompt.contains("<|system|>"));
        assert!(prompt.contains("<|user|>"));
        assert!(prompt.contains("<|assistant|>"));
    }

    #[test]
    fn test_token_count() {
        let provider = LocalProvider::new_random(TorusLLMConfig::tiny()).unwrap();
        let messages = vec![Message::user("Hello world!")];
        let count = provider.count_tokens(&messages).unwrap();
        assert!(count > 0);
    }
}
