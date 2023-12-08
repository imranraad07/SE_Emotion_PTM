import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer
import torch
from transformers import AutoModel
from torch.utils.data import Dataset, DataLoader
import nltk
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import re, sys, string, argparse, os
from transformers import get_linear_schedule_with_warmup
from transformers import BartTokenizer, BartForSequenceClassification, AdamW

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def load_data(train_file, test_file):
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(test_file)
    return train_df, val_df

def load_model(model_path):
    print(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    base_model = AutoModel.from_pretrained(
        model_path,
        output_attentions=True,
        output_hidden_states=True,
    )
    model = nn.Sequential(
        base_model,
        nn.Linear(base_model.config.hidden_size, 1),  # Only one output unit for binary classification
    )
    return model, tokenizer


def get_model_path(model_name):
    model_paths = {
        "bert": "bert-base-uncased",
        "roberta": "roberta-base",
        "deberta": "microsoft/deberta-v3-base",
        "albert": "albert-base-v2",
        "codebert": "microsoft/codebert-base",
        "graphcodebert": "microsoft/graphcodebert-base",
    }
    if model_name not in model_paths:
        raise ValueError("Invalid model_name. Choose from bert, codebert, graphcodebert, albert, deberta, and roberta.")
    return model_paths[model_name]

from nltk import pos_tag, RegexpParser
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn


def get_polarity_words(text):
    tokens = nltk.word_tokenize(text)
    pos_tags = nltk.pos_tag(tokens)

    polarity_words = []
    for word, pos in pos_tags:
        wn_pos = get_wordnet_pos(pos)
        synsets = wn.synsets(word, pos=wn_pos)
        if synsets:
            senti_synset = swn.senti_synset(synsets[0].name())
            if senti_synset.pos_score() > 0 or senti_synset.neg_score() > 0:
                polarity_words.append(word)

    return polarity_words

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wn.ADJ
    elif treebank_tag.startswith('V'):
        return wn.VERB
    elif treebank_tag.startswith('N'):
        return wn.NOUN
    elif treebank_tag.startswith('R'):
        return wn.ADV
    else:
        return None



def text_cleaning(text):
    text = str(text)
    printable = set(string.printable)
    text = ''.join(filter(lambda x: x in printable, text))
    text = text.replace('\x00', ' ')  # remove nulls
    text = text.replace('\r', ' ')
    text = text.replace('\n', ' ')
    text = text.lower()  # Lowercasing
    text = text.strip()
    # Tokenize and POS tagging
    tokens = nltk.word_tokenize(text)
    pos_tags = pos_tag(tokens)

    # Extract polarity words
    polarity_words = get_polarity_words(text)
    polarity_words = " ".join(polarity_words)
    
    return text, polarity_words


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
            text, polarity_words = text_cleaning(text)
            label = self.df.iloc[idx][self.col]

            # Tokenize the text and pad the sequences
            tokens = self.tokenizer.tokenize(text)
            tokens = tokens[:self.max_len - 2]
            tokens = ["[CLS]"] + tokens + ["[SEP]"]
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_ids = torch.nn.functional.pad(torch.tensor(input_ids), pad=(0, self.max_len - len(input_ids)), value=0)


            # Tokenize the text and pad the sequences
            polarity_tokens = self.tokenizer.tokenize(polarity_words)
            polarity_tokens = polarity_tokens[:self.max_len - 2]
            polarity_tokens = ["[CLS]"] + polarity_tokens + ["[SEP]"]
            polarity_input_ids = self.tokenizer.convert_tokens_to_ids(polarity_tokens)
            polarity_input_ids = torch.nn.functional.pad(torch.tensor(polarity_input_ids), pad=(0, self.max_len - len(polarity_input_ids)), value=0)

            # Convert the label to a tensor
            label = torch.tensor(label).long()

            return input_ids, polarity_input_ids, label
            # return input_ids, polarity_input_ids, label

    dataset = CSVDataset(df, tokenizer, max_len, col)
    return dataset

def train_model(model, train_dataloader, epochs, delta, optimizer, scheduler, val_dataloader, output_file, col, tokenizer):
    model.to(device)
    f1 = -0.1
    loss_val = 100
    criterion = nn.BCEWithLogitsLoss()  # Define the loss function for binary classification
    for epoch in range(epochs):
        total_train_loss = 0
        model.train()

        for step, batch in enumerate(train_dataloader):
            b_input_ids, b_polarity_input_ids, b_labels = batch
            b_input_ids = b_input_ids.to(device)
            b_polarity_input_ids = b_polarity_input_ids.to(device) 
            b_labels = b_labels.to(device).float().view(-1, 1)  # Change the shape to (batch_size, 1)

            model.zero_grad()

            attention_mask_input_ids = (b_input_ids != tokenizer.pad_token_id).float()
            attention_mask_polairy_input_ids = (b_polarity_input_ids != tokenizer.pad_token_id).float()
            
            # Forward pass through base model
            base_model_output = model[0](b_input_ids, attention_mask=attention_mask_input_ids)
            base_model_output_polarity = model[0](b_polarity_input_ids, attention_mask=attention_mask_polairy_input_ids)
                        
            last_hidden_state = base_model_output[0]  # Extract the last hidden state
            last_hidden_state_polairy = base_model_output_polarity[0]  # Extract the last hidden state            

            # Average the hidden states
            averaged_hidden_state = (last_hidden_state*0.75 + last_hidden_state_polairy*0.25)
            
            # Take the hidden state of the [CLS] token (first token)
#            cls_hidden_state = last_hidden_state[:, 0, :]
            cls_hidden_state = averaged_hidden_state[:, 0, :]

            # Apply the linear layer for classification
            logits = model[1](cls_hidden_state)  # Shape should be (batch_size, 1)

            # Calculate the loss
            loss = criterion(logits, b_labels)

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

        for batch in val_dataloader:
            b_input_ids, b_polarity_input_ids, b_labels = batch
            # b_input_ids, b_polarity_input_ids, b_labels = batch
            b_input_ids = b_input_ids.to(device)
            b_polarity_input_ids = b_polarity_input_ids.to(device) 
            b_labels = b_labels.to(device).float().view(-1, 1)  # Change the shape to (batch_size, 1)


            with torch.no_grad():
                attention_mask_input_ids = (b_input_ids != tokenizer.pad_token_id).float()
                attention_mask_polairy_input_ids = (b_polarity_input_ids != tokenizer.pad_token_id).float()
                
                # Forward pass through base model
                base_model_output = model[0](b_input_ids, attention_mask=attention_mask_input_ids)
                base_model_output_polarity = model[0](b_polarity_input_ids, attention_mask=attention_mask_polairy_input_ids)
                            
                last_hidden_state = base_model_output[0]  # Extract the last hidden state
                last_hidden_state_polairy = base_model_output_polarity[0]  # Extract the last hidden state            

                # Average the hidden states
                averaged_hidden_state = (last_hidden_state*0.75 + last_hidden_state_polairy*0.25)

                # Take the hidden state of the [CLS] token (first token)
#                cls_hidden_state = last_hidden_state[:, 0, :]
                cls_hidden_state = averaged_hidden_state[:, 0, :]
            
                # Apply the linear layer for classification
                logits = model[1](cls_hidden_state)  # Shape should be (batch_size, 1)

                logits = model[1](cls_hidden_state)
                loss = criterion(logits, b_labels)

            total_eval_loss += loss.item()

            # Apply sigmoid to the logits and round to get predictions
            preds = torch.sigmoid(logits).round().cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(b_labels.cpu().numpy())

        # Calculate the average loss over all of the batches
        avg_val_loss = total_eval_loss / len(val_dataloader)

        # Calculate the validation f1-score
        val_f1_score = f1_score(true_labels, predictions, average='binary')

        if avg_train_loss < delta:
            break

            
        if f1 < val_f1_score:
            print(f"Epoch {epoch + 1}: Average training loss: {avg_train_loss:.4f}, Average validation loss: {avg_val_loss:.4f}, Validation f1-score: {val_f1_score:.4f}")
            f1 = val_f1_score
            print(confusion_matrix(true_labels, predictions))
            print(classification_report(true_labels, predictions))
            my_array_pred = np.array(predictions).flatten()
            my_array_true = np.array(true_labels).flatten()
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

    output_file = model_name + '_polarity_' + col + '_' + output_file

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

    train_model(model, train_dataloader, epochs, delta, optimizer, scheduler, val_dataloader, output_file, col, tokenizer)

if __name__ == "__main__":
    main()

