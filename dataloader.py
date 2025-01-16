import torch
from torch.utils.data import DataLoader
from data import train_dataset, val_dataset, test_dataset, tokenizer
num_workers = 0
batch_size = 8

torch.manual_seed(123)

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    num_workers=num_workers,
    drop_last=False,
)
for batch in train_loader:
    # Access tensors from the batch
    input_ids = batch[0][0]  # First sequence in the batch
    labels = batch[1][0]     # Corresponding labels

    # Convert to lists if they are tensors
    input_ids = input_ids.tolist() if hasattr(input_ids, "tolist") else input_ids
    labels = labels.tolist() if hasattr(labels, "tolist") else labels

    # Ensure the inputs are flat lists
    if isinstance(input_ids[0], list):  # Check for nested lists
        input_ids = [item for sublist in input_ids for item in sublist]  # Flatten
    if isinstance(labels[0], list):  # Check for nested lists
        labels = [item for sublist in labels for item in sublist]  # Flatten

    # Decode the token IDs back to text
    decoded_input = tokenizer.decode(input_ids, skip_special_tokens=True)
    decoded_labels = tokenizer.decode(labels, skip_special_tokens=True)

    print("Decoded Input:", decoded_input)
    print("Decoded Labels:", decoded_labels)
    break
print(f"{len(train_loader)} training batches")
print(f"{len(val_loader)} validation batches")
print(f"{len(test_loader)} test batches")