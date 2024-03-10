from flask import Flask, render_template, request
from flask_socketio import SocketIO
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from gtts import gTTS
import os

app = Flask(__name__)
socketio = SocketIO(app)

def generate_response(input_text, model, tokenizer):
    try:
        if input_text.strip() == "":
            raise ValueError("Input text is empty. Please provide valid input.")

        # Tokenize the input text
        input_ids = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)['input_ids']

        if input_ids is None or len(input_ids) == 0:
            raise ValueError("Failed to tokenize input text. Please provide valid input.")

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

        return generated_text

    except Exception as e:
        return "Error: " + str(e)

def convert_text_to_speech(text):
    # Language in which you want to convert
    language = 'en'
    
    # Passing the text and language to the engine
    tts_object = gTTS(text=text, lang=language, slow=False)
    
    # Saving the converted audio in an mp3 file
    tts_object.save("response.mp3")
    
    # Playing the converted file
    os.system("mpg321 response.mp3")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('message')
def handle_message(data):
    user_input = data['message']
    response = generate_response(user_input, model, tokenizer)
    convert_text_to_speech(response)
    socketio.emit('response', {'message': response})

if __name__ == '__main__':
    # Load the fine-tuned GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("fine_tuned_therapist_model")
    model.eval()

    socketio.run(app, port=8002, debug=True)
