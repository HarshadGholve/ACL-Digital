import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Transformer

# Define the Transformer-based Text-to-SQL Model
class TextToSQLTransformer(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size, embedding_dim, num_heads, num_layers, hidden_dim, dropout=0.1):
        super(TextToSQLTransformer, self).__init__()
        
        # Embedding layer for the input sequence (question)
        # Converts input tokens (question words) to dense vectors of embedding_dim
        self.src_embedding = nn.Embedding(input_vocab_size, embedding_dim)

        # Embedding layer for the target sequence (SQL query)
        # Converts target tokens (SQL query words) to dense vectors of embedding_dim
        self.tgt_embedding = nn.Embedding(output_vocab_size, embedding_dim)

        # Transformer model: consists of an encoder and a decoder
        # encoder takes source embeddings (question),
        # decoder generates target embeddings (SQL query).
        self.transformer = Transformer(
            d_model=embedding_dim,  # Dimensionality of the model (embedding_dim)
            nhead=num_heads,        # Number of attention heads
            num_encoder_layers=num_layers,  # Number of encoder layers
            num_decoder_layers=num_layers,  # Number of decoder layers
            dim_feedforward=hidden_dim,  # Hidden dimension for the feedforward network
            dropout=dropout  # Dropout rate to avoid overfitting
        )

        # Fully connected layer to map the Transformer output to the output vocabulary size
        # This layer will produce the final SQL query tokens
        self.fc_out = nn.Linear(embedding_dim, output_vocab_size)

        # Positional Encoding to add information about the position of each token in the sequence
        # Helps the model understand the order of the input tokens
        self.positional_encoding = nn.Parameter(torch.zeros(1, 5000, embedding_dim))  # Max sequence length: 5000

    def forward(self, src, tgt):
        """
        Forward pass through the model
        :param src: Input sequence (question tokens) with shape (batch_size, seq_len)
        :param tgt: Target sequence (SQL query tokens) with shape (batch_size, seq_len)
        :return: Generated SQL query tokens
        """
        # Add positional encoding to the input and target embeddings
        # Positional encoding helps the model learn the order of tokens in the sequence
        src = self.src_embedding(src) + self.positional_encoding[:, :src.size(1), :]
        tgt = self.tgt_embedding(tgt) + self.positional_encoding[:, :tgt.size(1), :]

        # Pass through the Transformer model
        # The Transformer processes the question and generates the SQL query tokens
        output = self.transformer(src, tgt)

        # Apply the fully connected layer to the transformer output
        # This converts the transformer output into the desired SQL query vocabulary
        output = self.fc_out(output)

        # Return the output (predicted SQL query tokens)
        return output

# Function to create and initialize the model
def create_model(input_vocab_size, output_vocab_size, embedding_dim, num_heads, num_layers, hidden_dim):
    """
    Initializes and returns a TextToSQLTransformer model
    :param input_vocab_size: Vocabulary size for input questions
    :param output_vocab_size: Vocabulary size for output SQL queries
    :param embedding_dim: The dimension of the embedding vectors
    :param num_heads: Number of attention heads in the Transformer
    :param num_layers: Number of layers in the Transformer encoder and decoder
    :param hidden_dim: Dimension of the hidden layer in the feedforward network
    :return: Initialized TextToSQLTransformer model
    """
    model = TextToSQLTransformer(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        hidden_dim=hidden_dim
    )
    return model
