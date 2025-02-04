import torch
import gradio as gr
import yaml
import re

from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModel, AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

model_dir = config["model_dir"]
max_tokens = config["max_tokens"]
max_message_history = config["max_message_history"]

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

model = AutoModelForCausalLM.from_pretrained(model_dir, device_map = "auto", torch_dtype=torch.bfloat16, quantization_config=bit_config)

tokenizer = AutoTokenizer.from_pretrained(model_dir)

chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Initialize conversation history
conversation_history = [{"role": config["system_prompt_key"], "content": config["system_prompt"]}]

# Initialize a Conversation object
response = chatbot(conversation_history, max_new_tokens=max_tokens)
print("Please wait while the model is loading...")

print("---\nResponse:\n\n")
print(response[0]["generated_text"][-1]["content"])
print("\n---")

print("Model is ready!")

def vanilla_chatbot(message, history):
    global conversation_history
    
    # Add the new user input to the conversation history
    conversation_history.append({ "role": "user", "content": message})
    
    # Truncate the conversation history if it exceeds max_message_history
    if len(conversation_history) > max_message_history:
        # Keep the system prompt and the last 19 messages
        conversation_history = [conversation_history[0]] + conversation_history[-(max_message_history - 1):]
    
    # Generate a response
    response = chatbot(conversation_history, max_new_tokens=max_tokens, num_return_sequences=1)
    
    # Get the raw response text
    raw_response = response[0]["generated_text"][-1]
    
    # Clean the response using regex to remove thinking information
    thinking_pattern = r'[`\s]*[\[\<]think[\>\]](.*?)[\[\<]\/think[\>\]][`\s]*|^[`\s]*([\[\<]thinking[\>\]][`\s]*.*$)'
    cleaned_response = re.sub(thinking_pattern, '', raw_response['content'], flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    
    # Create the bot response object with the correct structure
    bot_response = {
        "role": "assistant",
        "content": cleaned_response.strip()
    }
    
    conversation_history.append(bot_response)
    
    return bot_response

demo_chatbot = gr.ChatInterface(vanilla_chatbot, type="messages", title="Vanilla Chatbot", description="Enter text to start chatting")

demo_chatbot.launch(server_name="0.0.0.0", share=False)