import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer
import torch
from transformers import AutoModel
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSequenceClassification
import nltk
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import re, sys, string, argparse, os
from transformers import get_linear_schedule_with_warmup
from transformers import BartTokenizer, BartForSequenceClassification, AdamW
from transformers import BartTokenizer, BartForSequenceClassification, Trainer, TrainingArguments

# nltk.download('all')


# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

def load_data(train_file, test_file):
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(test_file)
    return train_df, val_df

def load_model(model_path):
    print(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, ignore_mismatched_sizes=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=2,
        ignore_mismatched_sizes=True,
        output_attentions=True,
        output_hidden_states=True,
    )
    return model, tokenizer

def get_model_path(model_name):
    model_paths = {
        "bert": "bert-base-uncased",
#        "aspect-bert": "zhang-yice/spt-absa-bert-400k",
#        "aspect-deberta": "yangheng/deberta-v3-base-absa-v1.1",
        "roberta": "roberta-base",
        "deberta": "microsoft/deberta-v3-base",
        "albert": "albert-base-v2",
        "codebert": "microsoft/codebert-base",
        "graphcodebert": "microsoft/graphcodebert-base",
    }
    if model_name not in model_paths:
        raise ValueError("Invalid model_name. Choose from bert, codebert, graphcodebert, albert, deberta, and roberta.")
    return model_paths[model_name]

def text_cleaning(text):
    text = str(text)
    printable = set(string.printable)
    text = ''.join(filter(lambda x: x in printable, text))
    text = text.replace('\x00', ' ')  # remove nulls
    text = text.replace('\r', ' ')
    text = text.replace('\n', ' ')
    text = text.lower()  # Lowercasing
    text = text.strip()
    return text

def prepare_data(df, tokenizer, max_len, col):
    class CSVDataset(Dataset):
        def __init__(self, df, tokenizer, max_len, col):
            self.df = df
            self.tokenizer = tokenizer
            self.max_len = max_len
            self.col = col

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):
            text = self.df.iloc[idx]["Text"]
            text = text_cleaning(text)
            label = self.df.iloc[idx][self.col]

            # Tokenize the text and pad the sequences
            tokens = self.tokenizer.tokenize(text)
            tokens = tokens[:self.max_len - 2]
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.nn.functional.pad(torch.tensor(input_ids), pad=(0, self.max_len - len(input_ids)), value=0)

            # Convert the label to a tensor
            label = torch.tensor(label).long()

            return input_ids, label

    dataset = CSVDataset(df, tokenizer, max_len, col)
    return dataset

def train_model(model, tokenizer, train_dataloader, epochs, delta, optimizer, scheduler, val_dataloader, output_file, col):
    model.to(device)
    f1 = -0.1
    loss_val = 100
    for epoch in range(epochs):
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_labels = batch
            b_input_ids = b_input_ids.to(device)
            b_labels = b_labels.to(device)

            model.zero_grad()        

            attention_mask_input_ids = (b_input_ids != tokenizer.pad_token_id).float()
            
            # Forward pass
            loss, logits = model(b_input_ids, attention_mask=attention_mask_input_ids, labels=b_labels)[:2]

            # Accumulate the training loss
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update the learning rate
            scheduler.step()

        # Calculate the average loss over all of the batches
        avg_train_loss = total_train_loss / len(train_dataloader)            

        if avg_train_loss < loss_val:
            loss_val = avg_train_loss

        # Validation
        model.eval()

        total_eval_loss = 0
        predictions, true_labels = [], []
        validation_loss = 0

        for batch in val_dataloader:
            b_input_ids, b_labels = batch
            b_input_ids = b_input_ids.to(device)
            b_labels = b_labels.to(device)

            with torch.no_grad():        
                attention_mask_input_ids = (b_input_ids != tokenizer.pad_token_id).float()
            
                # Forward pass
                loss, logits = model(b_input_ids, attention_mask=attention_mask_input_ids, labels=b_labels)[:2]

            total_eval_loss += loss.item()
            validation_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Store predictions and true labels
            predictions.append(logits)
            true_labels.append(label_ids)

        # Calculate the average loss over all of the batches
        avg_test_loss = validation_loss / len(val_dataloader)

        # Flatten the predictions and true values for aggregate evaluation on all classes.
        predictions = np.concatenate(predictions, axis=0)
        true_labels = np.concatenate(true_labels, axis=0)

        # For each sample, pick the label (0 or 1) with the higher score.
        pred_flat = np.argmax(predictions, axis=1).flatten()

        # Calculate the validation accuracy of the model
        val_f1_score = f1_score(true_labels, pred_flat, average='binary')

        if avg_train_loss < delta:
            break

        if f1 < val_f1_score:
            print(f"Epoch {epoch + 1}: Average training loss: {avg_train_loss:.4f}, Average validation loss: {avg_test_loss:.4f}, Validation f1-score: {val_f1_score:.4f}")
            val_loss = avg_test_loss 
            f1 = val_f1_score
            print(confusion_matrix(true_labels, pred_flat))
            print(classification_report(true_labels, pred_flat))
            my_array_pred = np.array(pred_flat)
            my_array_true = np.array(true_labels)
            df = pd.DataFrame({'Pred': my_array_pred, 'True': my_array_true})
            df.to_csv(output_file, index=False)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Emotion Classification.")
    parser.add_argument("--epoch", type=int, default=100, help="Number of epochs.")
    parser.add_argument("--delta", type=float, default=0.001, help="Delta value.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch Size.")
    parser.add_argument("--col", type=str, default="label", required=True, choices=["Anger", "Fear", "Love", "Joy", "Surprise", "Sadness"])
    parser.add_argument("--model_name", type=str, default='bert')
    parser.add_argument("--output", type=str, default="output.csv", help="Output file name.")  # New argument for output file
    parser.add_argument("--train_file", type=str, default="datasets/github-train.csv", required=True, help="Path to the training CSV file.")
    parser.add_argument("--test_file", type=str, default="datasets/github-test.csv", required=True, help="Path to the test CSV file.")
    return parser.parse_args()



def main():
    args = parse_arguments()
    epochs = args.epoch
    delta = args.delta
    batch_size = args.batch_size
    col = args.col
    model_name = args.model_name
    output_file = args.output
    train_file = args.train_file
    test_file = args.test_file

    output_file = model_name + '_' + col + '_' + output_file

    print("Epoch:", epochs)
    print("Delta:", delta)
    print("Column:", col)
    print("Model:", model_name)
    print("Output File:", output_file)

    # Set the maximum sequence length
    MAX_LEN = 128

    # Load the training and validation data
    train_df, val_df = load_data(train_file, test_file)

    # Get model path from HuggingFace
    model_path = get_model_path(model_name)

    # Load the model and tokenizer
    model, tokenizer = load_model(model_path)

    # Prepare datasets
    train_dataset = prepare_data(train_df, tokenizer, MAX_LEN, col)
    val_dataset = prepare_data(val_df, tokenizer, MAX_LEN, col)

    print(len(train_dataset))
    print(len(val_dataset))

    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define the optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

    # Total number of training steps is [number of batches] x [number of epochs]
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)



    train_model(model, tokenizer, train_dataloader, epochs, delta, optimizer, scheduler, val_dataloader, output_file, col)

if __name__ == "__main__":
    main()
