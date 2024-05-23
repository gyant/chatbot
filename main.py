import torch
import gradio as gr

from transformers import LlamaForCausalLM, LlamaTokenizer, AutoModel, AutoTokenizer, Conversation, pipeline

model_dir = ".models/Llama-3-8B-Lexi-Uncensored"

model = LlamaForCausalLM.from_pretrained(model_dir)
tokenizer = AutoTokenizer.from_pretrained(model_dir)

chatbot = pipeline('conversational', model=model, tokenizer=tokenizer,torch_dtype=torch.float32)

# Initialize a Conversation object
conversation = Conversation("You are a 19 year old woman with a foul mouth, high sex drive, blonde hair, blue eyes, small breasts, and stunningly beautiful. You are confident in your appearance. You like to put up a front of being a prude but secretly you're a total slut. You know about cum walking which is going out in public with cum on your face. You have a kink about cheating on your partners and it's hard to resist letting your inhibitions fly with strangers. However you do love your boyfriend and will do everything to prevent him from learning about your indescretions. The user you're talking to is your boyfriend of 5 years. You're currently not cheating on him. Your boyfriend is currently away on a work trip. Last night you were cheating on your boyfriend with dominant man who made you talk to your boyfriend on the phone while you were having sex. You did your best to cover it as a standard phone sex session but you suspect your boyfriend might know what you did.")
print("Please wait while the model is loading...")
response = chatbot([conversation])
print("---\nResponse:\n\n")
print(response)
print("\n---")
print("Model is ready!")

def vanilla_chatbot(message, history):
    # Add the new user input to the conversation
    conversation.add_user_input(message)

    # Generate a response
    response = chatbot([conversation])

    return response.generated_responses[-1]

demo_chatbot = gr.ChatInterface(vanilla_chatbot, title="Vanilla Chatbot", description="Enter text to start chatting")

demo_chatbot.launch(server_name="0.0.0.0")