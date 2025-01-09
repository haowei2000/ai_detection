import torch
from datasets import Dataset
from transformers import BertTokenizer

# Sample dataset
data = {"text": ["Hello, world!", "How are you?"]}
dataset = Dataset.from_dict(data)

# Initialize tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Function to tokenize the text and convert to torch tensors


def tokenize_function(examples):
    # Tokenize the text
    encoding = tokenizer(
        examples["text"], padding=True, truncation=True, return_tensors="pt"
    )
    # Convert to pytorch tensors (they will already be tensors if using return_tensors="pt")
    return {key: torch.tensor(value) for key, value in encoding.items()}


# Apply the function with batched=True to process a batch at a time
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Check the result
print(tokenized_dataset["input_ids"])
