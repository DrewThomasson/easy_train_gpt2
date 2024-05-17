from transformers import GPT2Tokenizer, GPT2LMHeadModel

def load_model(model_path):
    # Load the fine-tuned model and tokenizer
    model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    return model, tokenizer

def generate_response(model, tokenizer, prompt):
    # Encode the prompt and generate response
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=200, num_return_sequences=1)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def main():
    # Load your fine-tuned model
    model_path = "output"  # Replace with your model directory path
    model, tokenizer = load_model(model_path)

    print("You can now talk to your fine-tuned model. Type 'quit' to exit.")
    
    while True:
        prompt = input("You: ")
        if prompt.lower() == "quit":
            break
        response = generate_response(model, tokenizer, prompt)
        print("Model:", response)

if __name__ == "__main__":
    main()
