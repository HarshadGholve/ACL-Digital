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
loss_fn = torch.nn.CrossEntropyLoss()

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids, output_ids = batch
        
        # Check the shape of input and output
        print(f"input_ids shape: {input_ids.shape}, output_ids shape: {output_ids.shape}")
        
        # Ensure both input_ids and output_ids have the same batch size
        assert input_ids.shape[0] == output_ids.shape[0], "Batch size mismatch between input and output"

        predictions = model(input_ids, output_ids)
        loss = loss_fn(predictions.transpose(1, 2), output_ids)
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
            predictions = model(input_ids)
            dev_loss = loss_fn(predictions.transpose(1, 2), output_ids)
            total_dev_loss += dev_loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Dev Loss: {total_dev_loss:.4f}")

# Save the model
torch.save(model.state_dict(), 'models/text_to_sql_transformer.pth')
print("Model saved!")
