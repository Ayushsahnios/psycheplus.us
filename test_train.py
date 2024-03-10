import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader

# Define a synthetic dataset class for demonstration purposes
class SyntheticTherapistPatientDataset(Dataset):
    def __init__(self):
        self.conversations = [
            "Therapist: How are you feeling today? Patient: I'm feeling a bit anxious.",
            "Therapist: What's been on your mind lately? Patient: I've been worrying about my job.",
            # Add more synthetic conversation examples as needed
        ]

        # Tokenize conversations
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.tokenized_conversations = [self.tokenizer.encode(conv) for conv in self.conversations]

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        return torch.tensor(self.tokenized_conversations[idx])

# Load pre-trained GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Fine-tune the model on synthetic therapist-patient conversation dataset
dataset = SyntheticTherapistPatientDataset()
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

model.train()
for epoch in range(3):  # Adjust number of epochs as needed
    for batch in dataloader:
        optimizer.zero_grad()
        input_ids = batch[:, :-1]
        labels = batch[:, 1:]
        outputs = model(input_ids, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Save the fine-tuned model
model.save_pretrained("fine_tuned_therapist_model")
