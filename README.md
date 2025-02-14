# Chatbot

A flexible chatbot implementation supporting both GGUF and SafeTensor models.

## Configuration Options

| Parameter | Type | Default | Description | Valid Values |
|-----------|------|---------|-------------|--------------|
| model_dir | string | - | Path to model directory or file | Path to .gguf file or HuggingFace model directory |
| model_type | string | "transformers" | Type of model to load | "gguf" or "transformers" |
| max_tokens | integer | 2048 | Maximum number of tokens in response | 1-4096 |
| max_message_history | integer | 10 | Maximum number of messages to keep in context | 2+ |
| system_prompt | string | - | Initial system prompt to set model behavior | Any text |
| system_prompt_key | string | "system" | Role key for system messages | Usually "system" |
| n_ctx | integer | 2048 | Context window size (GGUF only) | 512-4096 |
| n_threads | integer | 4 | Number of CPU threads (GGUF only) | 1-CPU core count |
| temperature | float | 0.7 | Randomness in response generation | 0.0-1.0 |

## Example Configuration

```
system_prompt: "you are a helpful weirdo."
system_prompt_key: "system"
model_dir: ./models/my-model
bit_config: eight
max_tokens: 256
max_message_history: 20
model_type: transformers
n_ctx: 2048
n_threads: 4
temperature: 0.7
```