import os
import torch
from transformers import AdamW
from data_processing import prepare_data
from model import TextToSQLTransformer

# Set paths for the dataset
train_path = 'data/raw/train_spider.json'
dev_path = 'data/raw/dev.json'

# Prepare data
train_loader, dev_loader, input_vocab, output_vocab = prepare_data(train_path, dev_path)

# Ensure the models directory exists
if not os.path.exists('models'):
    os.makedirs('models')

# Initialize the model
model = TextToSQLTransformer(
    input_vocab_size=len(input_vocab),
    output_vocab_size=len(output_vocab),
    embedding_dim=256,
    num_heads=8,
    num_layers=6,
    hidden_dim=512, 
    dropout=0.1
)

# Define optimizer
optimizer = AdamW(model.parameters(), lr=1e-4)

# Define loss function
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=0)

# # Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, output_ids = batch
        
        print(f"input_ids shape: {input_ids.shape}")
        print(f"output_ids shape: {output_ids.shape}")
        print(f"output_ids[:, :-1] shape: {output_ids[:, :-1].shape}")

        # Ensure tensors are in the right shape
        input_ids = input_ids.permute(1, 0)  # (batch_size, seq_len) -> (seq_len, batch_size)
        output_ids = output_ids.permute(1, 0)  # Same for output
        
        # Forward pass
        predictions = model(input_ids, output_ids[:, :-1])
        predictions = predictions.permute(1, 0, 2)  # Convert back to (batch_size, seq_len, feature_dim)

        # Compute loss
        loss = loss_fn(predictions.reshape(-1, predictions.size(-1)), output_ids[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

    # Evaluate on dev set
    model.eval()
    total_dev_loss = 0
    with torch.no_grad():
        for batch in dev_loader:
            input_ids, output_ids = batch

            # Ensure tensors are in the right shape
            input_ids = input_ids.permute(1, 0)  # (batch_size, seq_len) -> (seq_len, batch_size)
            output_ids = output_ids.permute(1, 0)  # Same for output

            predictions = model(input_ids, output_ids[:, :-1])
            predictions = predictions.permute(1, 0, 2)  # Convert back to (batch_size, seq_len, feature_dim)
            dev_loss = loss_fn(predictions.reshape(-1, predictions.size(-1)), output_ids[:, 1:].reshape(-1))
            total_dev_loss += dev_loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Dev Loss: {total_dev_loss:.4f}")

# Save the model
torch.save(model.state_dict(), 'models/text_to_sql_transformer.pth')
print("Model saved!")
