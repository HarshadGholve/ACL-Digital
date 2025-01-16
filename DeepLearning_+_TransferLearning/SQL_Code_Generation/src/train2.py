import pytorch_lightning as pl
from model2 import SQLModel
from data2 import SQLDataModule
import torch

def main():
    # Define paths to data
    train_path = 'data/raw/train_spider.json'
    dev_path = 'data/raw/dev.json'

    # Initialize the DataModule
    data_module = SQLDataModule(train_path, dev_path)

    # Initialize the model
    model = SQLModel(
        input_vocab_size=len(data_module.input_vocab),
        output_vocab_size=len(data_module.output_vocab),
        hidden_size=256,
        embedding_dim=100,
        max_len=50
    )

    # Initialize the Trainer
    trainer = pl.Trainer(max_epochs=1,devices=1)

    # Train the model
    trainer.fit(model, data_module)

if __name__ == '__main__':
    main()
