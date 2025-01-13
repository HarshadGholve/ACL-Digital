import torch
from model import TextToSQLTransformer
from data_processing import tokenize_input, detokenize_output
import sys

# Load the trained model
MODEL_PATH = 'models/text_to_sql_transformer.pth'

# Define the function to load the model
def load_model(input_vocab_size, output_vocab_size, embedding_dim=256, num_heads=8, num_layers=6, hidden_dim=512):
    model = TextToSQLTransformer(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()  # Set the model to evaluation mode
    return model

# Function to predict SQL query from natural language input
def predict(model, input_text, input_vocab, output_vocab):
    # Tokenize the input text
    tokenized_input = tokenize_input(input_text, input_vocab)
    input_tensor = torch.tensor([tokenized_input])

    # Create a start token for the target sequence
    start_token = output_vocab['<START>']
    tgt_tensor = torch.tensor([[start_token]])

    # Generate the SQL query
    with torch.no_grad():
        output = model(input_tensor, tgt_tensor)

    # Convert output tokens to SQL query
    output_tokens = torch.argmax(output, dim=-1).squeeze().tolist()
    sql_query = detokenize_output(output_tokens, output_vocab)
    
    return sql_query

# Main function to take user input and generate SQL query
if __name__ == "__main__":
    # Check if input is provided
    if len(sys.argv) != 2:
        print("Usage: python predict.py '<your question here>'")
        sys.exit(1)

    # User input
    user_input = sys.argv[1]

    # Load vocabularies
    input_vocab = torch.load('data/vocab/input_vocab.pth')
    output_vocab = torch.load('data/vocab/output_vocab.pth')

    # Load the model
    model = load_model(len(input_vocab), len(output_vocab))

    # Predict SQL query
    sql_query = predict(model, user_input, input_vocab, output_vocab)
    print("Generated SQL Query:")
    print(sql_query)
