import torch
import gradio as gr
import yaml
import re
from llama_cpp import Llama
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

model_dir = config["model_dir"]
max_tokens = config["max_tokens"]
max_message_history = config["max_message_history"]
model_type = config.get("model_type", "transformers")
n_ctx = config.get("n_ctx", 2048)
n_threads = config.get("n_threads", 4)
temperature = config.get("temperature", 0.7)
listen_host = config.get("listen_host", "0.0.0.0")
listen_port = config.get("listen_port", 7860)
verbose = config.get("verbose", False)

four_bit_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

eight_bit_config = BitsAndBytesConfig(
    load_in_8bit=True,
)

if config["bit_config"] == "four":
    bit_config = four_bit_config
elif config["bit_config"] == "eight":
    bit_config = eight_bit_config
elif config["bit_config"] == "sixteen":
    bit_config = None
else:
    bit_config = None

# Initialize model based on type
if model_type == "gguf":
    # GGUF model initialization with GPU support
    model = Llama(
        model_path=model_dir,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=-1,  # Use all layers on GPU (-1) or specify number of layers
        main_gpu=0,       # Main GPU device to use (0 = first GPU)
        tensor_split=None  # Optional: Split layers across multiple GPUs
    )
else:
    # Original SafeTensor initialization
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, 
        device_map="auto", 
        torch_dtype=torch.bfloat16, 
        quantization_config=bit_config
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Initialize conversation history
conversation_history = [{"role": config["system_prompt_key"], "content": config["system_prompt"]}]

# Remove thinking pattern from response in deepseek style models
THINKING_PATTERN = re.compile(
    r'[`\s]*[\[\<]think[\>\]](.*?)[\[\<]\/think[\>\]][`\s]*|^[`\s]*([\[\<]thinking[\>\]][`\s]*.*$)',
    flags=re.IGNORECASE | re.MULTILINE | re.DOTALL
)

# Initialize a Conversation object
if model_type == "gguf":
    response = model.create_chat_completion(
        messages=conversation_history,
        max_tokens=max_tokens,
        temperature=temperature
    )
    print("Please wait while the model is loading...")
    print("---\nResponse:\n\n")
    print(response["choices"][0]["message"]["content"])
else:
    response = text_generator(conversation_history, max_new_tokens=max_tokens)
    print("Please wait while the model is loading...")
    print("---\nResponse:\n\n")
    print(response[0]["generated_text"][-1]["content"])

print("\n---")

print("Model is ready!")

def vanilla_chatbot(message, history):
    global conversation_history
    
    # Add the new user input to the conversation history
    conversation_history.append({"role": "user", "content": message})

    if model_type == "gguf":
        # Let the model handle context window management
        response = model.create_chat_completion(
            messages=[{"role": m["role"], "content": m["content"]} for m in conversation_history],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        raw_response = response["choices"][0]["message"]["content"]

        if verbose:
            print(f"Raw response:\n\n{raw_response}\n\n")

        cleaned_response = THINKING_PATTERN.sub('', raw_response).strip()
   
    else:   
        # Original SafeTensor generation
        response = text_generator(conversation_history, max_new_tokens=max_tokens, num_return_sequences=1)
        raw_response = response[0]["generated_text"][-1]

        if verbose:
            print(f"Raw response:\n\n{raw_response}\n\n")
    
        cleaned_response = THINKING_PATTERN.sub('', raw_response["content"]).strip()
    
    # Create and append bot response
    bot_response = {
        "role": "assistant",
        "content": cleaned_response
    }

    conversation_history.append(bot_response)

    # Truncate history for transformers models
    if len(conversation_history) > max_message_history:
        conversation_history = [conversation_history[0]] + conversation_history[-(max_message_history - 1):]
    
    return bot_response

chat_interface = gr.ChatInterface(vanilla_chatbot, type="messages", title="Vanilla Chatbot", description="Enter text to start chatting")

chat_interface.launch(server_name=listen_host, server_port=listen_port, share=False)
