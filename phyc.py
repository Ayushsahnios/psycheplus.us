from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load the GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Input text to be tokenized
input_text = "How's your day going?"

try:
    if input_text.strip() == "":
        raise ValueError("Input text is empty. Please provide valid input.")

    # Tokenize the input text
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)['input_ids']
    
    if input_ids is None or len(input_ids) == 0:
        raise ValueError("Failed to tokenize input text. Please provide valid input.")

    print("Tokenized input:", input_ids)

    # Create attention mask to exclude padding tokens
    attention_mask = torch.ones_like(input_ids)
    attention_mask[input_ids == tokenizer.pad_token_id] = 0

    # Generate text based on the tokenized input
    output = model.generate(
        input_ids,
        max_length=100,  # Increased maximum length for more diverse responses
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,  # Set pad_token_id to eos_token_id
        attention_mask=attention_mask,  # Attention mask to exclude padding tokens
        temperature=0.7,  # Adjust temperature to control randomness
        top_k=50,  # Adjust top_k to control diversity
        top_p=0.95,  # Adjust top_p to control diversity
        do_sample=True  # Set do_sample to True to enable sample-based generation
    )

    # Decode the generated token IDs back into text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Post-processing: Remove repetitive text
    generated_text = ' '.join(dict.fromkeys(generated_text.split()))

    # Print the generated text
    print("Generated text:", generated_text)

except Exception as e:
    print("Error:", e)
