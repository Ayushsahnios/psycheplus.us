from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # Adding padding token
model = GPT2LMHeadModel.from_pretrained(model_name)

# Tokenize and encode preprocessed data
input_texts = []
target_texts = []
with open('preprocessed_data.txt', 'r') as f:
    for line in f:
        input_text, target_text = line.strip().split('\t')
        input_texts.append(input_text)
        target_texts.append(target_text)

input_encodings = tokenizer(input_texts, return_tensors='pt', padding=True, truncation=True)
target_encodings = tokenizer(target_texts, return_tensors='pt', padding=True, truncation=True)

# Ensure input_ids and labels are not outside the range of the vocabulary
input_encodings['input_ids'][input_encodings['input_ids'] >= tokenizer.vocab_size] = tokenizer.pad_token_id
target_encodings['input_ids'][target_encodings['input_ids'] >= tokenizer.vocab_size] = tokenizer.pad_token_id

# Prepare training data
train_dataset = torch.utils.data.TensorDataset(input_encodings.input_ids, target_encodings.input_ids)

# Fine-tune the model
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)

for epoch in range(3):
    for input_ids, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Save the fine-tuned model
model.save_pretrained('fine_tuned_model')
tokenizer.save_pretrained('fine_tuned_model')
