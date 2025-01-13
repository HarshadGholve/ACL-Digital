import json
import os
import nltk
import pandas as pd
from collections import Counter
from nltk.tokenize import word_tokenize
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Ensure that the required NLTK resources are available
nltk.download('punkt')

# Define file paths
train_file = 'data/raw/train_spider.json'
dev_file = 'data/raw/dev.json'

# Function to load the dataset from a JSON file
def load_data(file_path):
    """
    Load JSON data from the given file path.
    :param file_path: Path to the JSON dataset
    :return: Loaded data in JSON format
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

# Load training and dev datasets
train_data = load_data(train_file)
dev_data = load_data(dev_file)

print(f"Loaded {len(train_data)} training examples.")
print(f"Loaded {len(dev_data)} dev examples.")

# Function to extract questions, SQLs, and databases from the data
def extract_data(data):
    questions = []
    sql_queries = []
    databases = []

    for example in data:
        questions.append(example.get('question', ''))
        sql_queries.append(example.get('sql', ''))
        databases.append(example.get('database', 'unknown'))

    return pd.DataFrame({
        'question': questions,
        'sql': sql_queries,
        'database': databases
    })

# Prepare training and dev data
train_df = extract_data(train_data)
dev_df = extract_data(dev_data)

# Schema Integration

def load_schema(database_name):
    schema_file = f'data/raw/database/{database_name}/schema.sql'
    if os.path.exists(schema_file):
        with open(schema_file, 'r') as f:
            schema = f.read()
        return schema
    return None

# Example for loading restaurant schema
restaurant_schema = load_schema('restaurants')
print(restaurant_schema[:500])  # Preview the first 500 characters of the schema

# Function to build vocabulary
def build_vocab(data):
    counter = Counter()

    for example in data:
        # Tokenizing the question
        question = example.get('question', '')
        question_tokens = nltk.word_tokenize(question.lower())
        counter.update([token.lower() for token in question_tokens])

        # Tokenizing the query field (SQL query)
        sql_query = example.get('query', '')
        if isinstance(sql_query, str):  # Ensure that query is a string before tokenizing
            sql_query_tokens = nltk.word_tokenize(sql_query.lower())
            counter.update([token.lower() for token in sql_query_tokens])

        # Handling the structured SQL field ('sql' part)
        sql_struct = example.get('sql', {})

        # Tokenize 'select' part
        if 'select' in sql_struct:
            select_tokens = str(sql_struct['select']).split()  # Convert to string and split
            counter.update([token.lower() for token in select_tokens])

        # Tokenize 'from' part
        if 'from' in sql_struct:
            from_tokens = str(sql_struct['from']).split()  # Convert to string and split
            counter.update([token.lower() for token in from_tokens])

        # Tokenize 'where' part
        if 'where' in sql_struct:
            where_tokens = str(sql_struct['where']).split()
            counter.update([token.lower() for token in where_tokens])

        # Tokenize other SQL parts: groupBy, orderBy, having, limit, etc.
        for key in ['groupBy', 'orderBy', 'having']:
            if key in sql_struct:
                tokens = str(sql_struct[key]).split()
                counter.update([token.lower() for token in tokens])

        # Tokenize 'limit' part
        if 'limit' in sql_struct and sql_struct['limit'] is not None:
            counter.update([str(sql_struct['limit']).lower()])

        # Tokenize logical operators: union, intersect, except
        for key in ['union', 'intersect', 'except']:
            if sql_struct.get(key) is not None:
                counter.update([key.lower()])

    # Create vocabulary with token to index mapping
    vocab = {word: idx for idx, (word, _) in enumerate(counter.items(), 1)}
    vocab['<pad>'] = 0  # Padding token at index 0
    return vocab

def tokenize_input(input_text):
    """
    Tokenizes a natural language input into a list of tokens.
    :param input_text: Natural language question
    :return: List of tokens
    """
    return nltk.word_tokenize(input_text.lower())

# Detokenize output SQL query
def detokenize_output(output_tokens):
    """
    Detokenizes a list of SQL query tokens into a string.
    :param output_tokens: List of tokens representing SQL query
    :return: SQL query as a string
    """
    return ' '.join(output_tokens)


# Function to tokenize the questions and SQL queries
def tokenize_data(data, input_vocab, output_vocab):
    input_tokens = []
    output_tokens = []

    for example in data:
        # Tokenize the question (input)
        input_question = nltk.word_tokenize(example['question'].lower())
        # Convert tokens to indices based on the input vocabulary
        input_indices = [input_vocab.get(token, input_vocab.get('<pad>')) for token in input_question]
        input_tokens.append(input_indices)

        # Handle the 'query' field for the SQL query (output)
        output_query = example.get('query', '').lower()
        output_indices = [output_vocab.get(token, output_vocab.get('<pad>')) for token in nltk.word_tokenize(output_query)]
        output_tokens.append(output_indices)

    return input_tokens, output_tokens

# PyTorch Dataset class to handle tokenized data
class SQLDataset(Dataset):
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

# Custom collate function for padding
def collate_fn(batch):
    inputs, outputs = zip(*batch)

    # Convert to tensors, ensuring that each element is a tensor of integers
    inputs = [torch.tensor(inp) for inp in inputs]
    outputs = [torch.tensor(out) for out in outputs]

    # Pad sequences to the same length (using <pad> index as padding value)
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)  # 0 is the <pad> index
    padded_outputs = pad_sequence(outputs, batch_first=True, padding_value=0)  # 0 is the <pad> index

    return padded_inputs, padded_outputs

# Function to prepare the data for training and development
def prepare_data(train_path, dev_path):
    """
    Prepares the data for training and development.
    Loads the data, tokenizes it, and creates dataloaders.
    """
    train_data = load_data(train_path)
    dev_data = load_data(dev_path)

    # Build vocabularies from training data
    input_vocab = build_vocab(train_data)
    output_vocab = build_vocab(train_data)

    # Tokenize data using the vocabularies
    train_input, train_output = tokenize_data(train_data, input_vocab, output_vocab)
    dev_input, dev_output = tokenize_data(dev_data, input_vocab, output_vocab)

    # Create datasets for PyTorch DataLoader
    train_dataset = SQLDataset(train_input, train_output)
    dev_dataset = SQLDataset(dev_input, dev_output)

    # Create DataLoader with batching and shuffling
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=32, collate_fn=collate_fn)

    # Save vocabularies to files
    os.makedirs('data/vocab', exist_ok=True)  # Create folder if it doesn't exist
    torch.save(input_vocab, 'data/vocab/input_vocab.pth')
    torch.save(output_vocab, 'data/vocab/output_vocab.pth')


    return train_loader, dev_loader, input_vocab, output_vocab

# Example usage
if __name__ == "__main__":
    train_loader, dev_loader, input_vocab, output_vocab = prepare_data(
        'data/raw/train_spider.json',
        'data/raw/dev.json'
    )

    # Print example batch to verify DataLoader
    for batch in train_loader:
        inputs, outputs = batch
        print(f"Inputs shape: {inputs.shape}, Outputs shape: {outputs.shape}")
        break  # Just printing the first batch for verification
