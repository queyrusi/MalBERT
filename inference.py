import torch
from torch.utils.data import Dataset, DataLoader
from transformers import LongformerForSequenceClassification, LongformerTokenizer
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import warnings
import datetime
import argparse
from tqdm import tqdm

# --- Environment and writer

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('model_path', help='Path to the model file')
parser.add_argument('test_data_path', help='Path to the test data file')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
parser.add_argument('--logdir', default='logs/inferences', help='Directory to save inference logs')
args = parser.parse_args()

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

label_correspondence = {"goodware": 0, "malware": 1}

# Create the logs/inferences directory if it doesn't exist
logdir = args.logdir
os.makedirs(logdir, exist_ok=True)

# Get the current datetime
current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Create the log file path
log_file_path = os.path.join(logdir, f"{current_datetime}.log")

def confusion_matrix(y_true, y_pred):
    # Convert lists to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Compute confusion matrix
    TP = np.sum(np.where((y_true == 1) & (y_pred == 1), 1, 0))
    TN = np.sum(np.where((y_true == 0) & (y_pred == 0), 1, 0))
    FP = np.sum(np.where((y_true == 0) & (y_pred == 1), 1, 0))
    FN = np.sum(np.where((y_true == 1) & (y_pred == 0), 1, 0))

    return TP, TN, FP, FN

def print_scores(TP, TN, FP, FN):
    print(f"Test Results over {len(test_dataset)} samples")
    print("True Positives:", TP)
    print("True Negatives:", TN)
    print("False Positives:", FP)
    print("False Negatives:", FN)
    try:
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1_score = 2 * TP / (2 * TP + FP + FN)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1_score)
    except ZeroDivisionError:
        print("Error: Division by zero occurred.")
   
def log_confusion_matrix(TP, TN, FP, FN,batch_number, log_file):
    # Open the log file in append mode
    with open(log_file_path, 'a') as log_file:
        # Write the batch number and the current values of total_TP, total_TN, etc.
        log_file.write(f"Batch Number: {batch_number} - TP: {TP} TN: {TN} FP: {FP} FN: {FN}")
        log_file.write("\n")

# --- Data and model preparation

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

# Load the test data
test_data_path = 'data/test/D0'
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
test_dataset = CustomDataset(test_data_path, tokenizer)
test_loader = DataLoader(test_dataset, batch_size=2)

# Load the trained model
model_state_dict = torch.load(args.model_path)
model = LongformerForSequenceClassification.from_pretrained('allenai/longformer-base-4096', num_labels=2)
model.load_state_dict(model_state_dict)
model = model.to(device)
model.eval()

total_TP, total_TN, total_FP, total_FN = 0, 0, 0, 0

# --- Inference

# Iterate over the test data and make predictions
for batch_number, batch in enumerate(tqdm(test_loader)):
    inputs, labels = batch['input_ids'], batch['labels']

    # Move the inputs and labels to the device
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Make predictions
    with torch.no_grad():
        outputs = model(inputs)  # LongformerSequenceClassifierOutput(loss=None, logits=tensor([[ 0.2081, -0.0536]])

        # Get the predicted class
        _, predicted_class = torch.max(outputs.logits, 1)  # tensor([0, 0, 1, ...])

        # Convert tensors to lists for use with sklearn
        labels_list = labels.tolist()
        predicted_class_list = predicted_class.tolist()

        # Extract TP, TN, FP, FN
        TN, FP, FN, TP = confusion_matrix(labels_list, predicted_class_list)

        # Log conf mat of this batch
        log_confusion_matrix(TP, TN, FP, FN, batch_number, log_file_path)

        # Update the total values
        total_TP += TP; total_TN += TN; total_FP += FP; total_FN += FN
    
# Print the final scores
print_scores(total_TP, total_TN, total_FP, total_FN)