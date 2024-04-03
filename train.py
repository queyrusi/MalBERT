import os
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerTokenizer, LongformerForSequenceClassification, AdamW
import warnings

warnings.filterwarnings("ignore")

# --- Environment

device = "cuda" if torch.cuda.is_available() else "cpu"

# Set the seed for reproducibility
random.seed(42)
torch.manual_seed(42)

# --- Data Preparation

label_correspondence = {"goodware": 0, "malware": 1}

# Define the dataset class
class CustomDataset(Dataset):
    def __init__(self, data_dir, tokenizer):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.labels = []
        self.sentences = []
        self.load_data()

    def load_data(self):
        for label in os.listdir(self.data_dir):
            label_dir = os.path.join(self.data_dir, label)
            for file_name in os.listdir(label_dir):
                file_path = os.path.join(label_dir, file_name)
                with open(file_path, 'r') as file:
                    sentence = file.read().strip()
                    self.labels.append(int(label_correspondence[label]))
                    self.sentences.append(sentence)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inputs = self.tokenizer.encode_plus(
            self.sentences[idx],
            add_special_tokens=True,
            max_length=1024,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': torch.tensor(self.labels[idx])
        }

# Set the paths to your data directories
train_data_dir = 'data/train/D0'
# test_data_dir = 'data/test/D0'

# Initialize the tokenizer and model
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', num_labels=2)
model.to(device)

# Create the datasets
train_dataset = CustomDataset(train_data_dir, tokenizer)
# test_dataset = CustomDataset(test_data_dir, tokenizer)

# Split the train dataset into train and validation
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])

# Create the data loaders
train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
# test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Set the optimizer and learning rate
optimizer = AdamW(model.parameters(), lr=1e-5)

# --- Training

for epoch in range(5):
    model.train()
    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            val_loss += outputs.loss.item()
            _, predicted = torch.max(outputs.logits, dim=1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_accuracy = val_correct / val_total
    print(f'Epoch {epoch+1}: Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

# --- Testing

# Evaluate on the test set
# model.eval()
# test_loss = 0
# test_correct = 0
# test_total = 0
# with torch.no_grad():
#     for batch in test_dataloader:
#         input_ids = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         labels = batch['labels'].to(device)
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
#         test_loss += outputs.loss.item()
#         _, predicted = torch.max(outputs.logits, dim=1)
#         test_total += labels.size(0)
#         test_correct += (predicted == labels).sum().item()

# test_accuracy = test_correct / test_total
# print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')
