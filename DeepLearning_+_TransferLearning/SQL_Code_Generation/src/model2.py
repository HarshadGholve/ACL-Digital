import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SQLModel(pl.LightningModule):
    def __init__(self, input_vocab_size, output_vocab_size, hidden_size=256, embedding_dim=100, max_len=50):
        super(SQLModel, self).__init__()

        # Hyperparameters
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.max_len = max_len

        # Model layers
        self.input_embedding = nn.Embedding(self.input_vocab_size, self.embedding_dim)
        self.output_embedding = nn.Embedding(self.output_vocab_size, self.embedding_dim)
        
        self.encoder = nn.LSTM(self.embedding_dim, self.hidden_size, batch_first=True)
        self.decoder = nn.LSTM(self.embedding_dim, self.hidden_size, batch_first=True)
        
        self.fc = nn.Linear(self.hidden_size, self.output_vocab_size)
        
        # Store the outputs for validation
        self.val_outputs = []

    def forward(self, input_seq, output_seq=None, teacher_forcing_ratio=0.5):
        # Input Embedding
        embedded_input = self.input_embedding(input_seq)
        
        # Encoder
        packed_input = pack_padded_sequence(embedded_input, lengths=torch.sum(input_seq != 0, dim=1), batch_first=True, enforce_sorted=False)
        _, (hidden, cell) = self.encoder(packed_input)
        
        # Decoder
        decoder_input = self.output_embedding(output_seq) if output_seq is not None else torch.zeros_like(embedded_input)
        packed_output, _ = self.decoder(decoder_input, (hidden, cell))
        unpacked_output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Output Layer
        logits = self.fc(unpacked_output)
        
        return logits

    def training_step(self, batch, batch_idx):
        input_seq, output_seq = batch
        output_seq_input = output_seq[:, :-1]  # Ignore last token for teacher forcing
        logits = self(input_seq, output_seq_input)
        
        # Calculate the loss
        loss = nn.CrossEntropyLoss(ignore_index=0)(logits.view(-1, self.output_vocab_size), output_seq[:, 1:].contiguous().view(-1))
        return loss

    def validation_step(self, batch, batch_idx):
        input_seq, output_seq = batch
        output_seq_input = output_seq[:, :-1]
        logits = self(input_seq, output_seq_input)
        
        loss = nn.CrossEntropyLoss(ignore_index=0)(logits.view(-1, self.output_vocab_size), output_seq[:, 1:].contiguous().view(-1))
        
        # Store validation output for on_validation_epoch_end
        self.val_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        """
        Replaces validation_epoch_end in Lightning v2.0.0.
        Computes the average validation loss after the epoch.
        """
        if len(self.val_outputs) > 0:
            avg_val_loss = torch.stack(self.val_outputs).mean()
            self.log("validation_loss", avg_val_loss)
        
        # Clear the stored outputs after processing
        self.val_outputs.clear()  # Free memory

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
