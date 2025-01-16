import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import json
import nltk
import os
from urllib import request
from torch.nn.utils.rnn import pad_sequence

nltk.download('punkt')

class SQLDataset(Dataset):
    def __init__(self, inputs, outputs, input_vocab, output_vocab, max_len):
        self.inputs = [self.pad_sequence([input_vocab[token] for token in inp], max_len) for inp in inputs]
        self.outputs = [self.pad_sequence([output_vocab[token] for token in out], max_len) for out in outputs]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx], dtype=torch.long), torch.tensor(self.outputs[idx], dtype=torch.long)

    def pad_sequence(self, seq, max_len, pad_token=0):
        return seq + [pad_token] * (max_len - len(seq))

class SQLDataModule(pl.LightningDataModule):
    def __init__(self, train_path, dev_path, batch_size=32, max_len=50):
        super(SQLDataModule, self).__init__()
        self.train_path = train_path
        self.dev_path = dev_path
        self.batch_size = batch_size
        self.max_len = max_len
        self.input_vocab = defaultdict(lambda: len(self.input_vocab))  # Initialize the vocab dictionary
        self.output_vocab = defaultdict(lambda: len(self.output_vocab))  # Initialize the vocab dictionary

    def prepare_data(self):
        """
        A one-time function to prepare data. It could involve downloading the datasets
        and saving them to local paths if they do not exist.
        """
        print("Preparing data (downloading if necessary)")

        # # Check if the data already exists locally, if not, download it
        # if not os.path.exists(self.train_path):
        #     print(f"Training data not found at {self.train_path}. Downloading...")
        #     self.download_data("https://example.com/train_data.json", self.train_path)

        # if not os.path.exists(self.dev_path):
        #     print(f"Development data not found at {self.dev_path}. Downloading...")
        #     self.download_data("https://example.com/dev_data.json", self.dev_path)
        pass

    def download_data(self, url, save_path):
        """
        Helper function to download data from the URL and save it locally.
        """
        try:
            request.urlretrieve(url, save_path)
            print(f"Data downloaded and saved to {save_path}")
        except Exception as e:
            print(f"Error downloading data: {e}")
            raise

    def setup(self, stage=None):
        # Load data from JSON
        def load_data(file_path):
            with open(file_path, 'r') as f:
                data = json.load(f)
            inputs = [entry['question_toks'] for entry in data]
            outputs = [entry['query_toks'] for entry in data]
            return inputs, outputs

        train_inputs, train_outputs = load_data(self.train_path)
        dev_inputs, dev_outputs = load_data(self.dev_path)

        # Add tokens to vocab
        for question in train_inputs:
            for token in question:
                self.input_vocab[token]  # Add token to the vocab
        for query in train_outputs:
            for token in query:
                self.output_vocab[token]  # Add token to the vocab

        # Special tokens
        self.input_vocab['<pad>'] = 0
        self.output_vocab['<pad>'] = 0

        # Create datasets
        self.train_dataset = SQLDataset(train_inputs, train_outputs, self.input_vocab, self.output_vocab, self.max_len)
        self.dev_dataset = SQLDataset(dev_inputs, dev_outputs, self.input_vocab, self.output_vocab, self.max_len)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.batch_size)
