# Emotion Classification Script

This script performs emotion classification using various transformer-based models such as BERT, RoBERTa, ALBERT, and CodeBERT. It uses PyTorch and the Hugging Face Transformers library to build and train the classification model. The script supports both traditional and contrastive fine-tuning approaches for the models.

## Requirements

To run this script, you need the following dependencies:

- Python 3.x
- PyTorch
- Transformers library (Hugging Face)
- pandas
- nltk

Install the required packages using the following command:

`pip install torch transformers pandas nltk`



## Usage

1. Make sure you have training and test data in CSV format. The CSV files should contain two columns: "text" (containing the text to classify) and "label" (containing the class labels). The label value should be Anger, Fear, Love, Joy, Surprise, Sadness. This is a multi-label dataset.


2. Run the emotion classification script with the following command:


`python emotion_bert.py --epoch EPOCH --delta DELTA --batch_size BATCH_SIZE --col COLUMN --model_name MODEL_NAME --output OUTPUT_FILE --train_file TRAIN_CSV --test_file TEST_CSV`

OR

`python emotion_bert_polarity.py --epoch EPOCH --delta DELTA --batch_size BATCH_SIZE --col COLUMN --model_name MODEL_NAME --output OUTPUT_FILE --train_file TRAIN_CSV --test_file TEST_CSV`

### Arguments:

- `EPOCH`: Number of epochs for training (default: 100).
- `DELTA`: Early stopping criterion. Training will stop if the average training loss goes below this value (default: 0.01).
- `BATCH_SIZE`: Batch size for training (default: 64).
- `COLUMN`: The emotion column to classify (choose from Anger, Fear, Love, Joy, Surprise, Sadness) (required).
- `MODEL_NAME`: Model architecture to use (choose from bert, roberta, albert, codebert, graphcodebert, deberta) (required).
- `OUTPUT_FILE`: Output file name for storing predictions (default: output.csv).
- `TRAIN_CSV`: Path to the training CSV file (required).
- `TEST_CSV`: Path to the test CSV file (required).

## Output

The script will print the training progress, average training loss, average validation loss, and validation F1-score for each epoch. Additionally, it will save the predictions in the specified output file in CSV format, containing columns "Pred" (predicted labels) and "True" (true labels).
