import os
import pandas as pd
import re
from torch.utils.data import Dataset
from typing import Union, List, Dict, Any
import kagglehub
import torch 
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
path = kagglehub.dataset_download("jonathanmoore2/trumps-tweets-upto-dec-3rd-2020")

data_dir = path #'/root/.cache/kagglehub/datasets/jonathanmoore2/trumps-tweets-upto-dec-3rd-2020/versions/1'

# Find the CSV file within the directory
for filename in os.listdir(data_dir):
    if filename.endswith(".csv"):
        csv_file_path = os.path.join(data_dir, filename)
        break  # Stop after finding the first CSV file

# Read the CSV file
df = pd.read_csv(csv_file_path)
df = df[df['isRetweet'] != 't']
df = df.reset_index(drop=True) # reset index

tweets = df.copy()
tweets['text'] = tweets['text'].str.replace("&amp", "&")
tweets['text'] = tweets['text'].str.replace("&amp;", "&")
def remove_links(text):
  """Removes links from a given text string."""
  return re.sub(r'http\S+', '', text)
tweets['text'] = tweets['text'].apply(remove_links)
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

# Ensure the ratios sum to 1
assert train_ratio + val_ratio + test_ratio == 1.0, "Split ratios must sum to 1."

# Split the dataset
train_data, temp_data = train_test_split(df, test_size=(val_ratio + test_ratio), random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=test_ratio / (val_ratio + test_ratio), random_state=42)

# Save the splits to separate files
train_data.to_csv("tweets_train.csv", index=False)
val_data.to_csv("tweets_val.csv", index=False)
test_data.to_csv("tweets_test.csv", index=False)

print("Data split completed.")
print(f"Training set: {len(train_data)} samples")
print(f"Validation set: {len(val_data)} samples")
print(f"Test set: {len(test_data)} samples")
class DatasetV1(Dataset):
    def __init__(self,dataset,tokenizer,max_length,stride):

        self.data = pd.read_csv(dataset) 
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["text"].astype(str)]
        self.max_length = max_length or self._longest_encoded_length()
        self.pad_token_id = 50256
        self.stride = stride
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            # Truncate sequences if they are longer than max_length
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # Pad sequences to the longest sequence
        self.encoded_texts = [
            encoded_text + [self.pad_token_id] * (self.max_length - len(encoded_text))
            for encoded_text in self.encoded_texts
        ]
        self.input_ids = []
        self.target_ids = []

     
      

        # Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(self.encoded_texts) - max_length, stride):
            input_chunk = self.encoded_texts[i:i + max_length]
            target_chunk = self.encoded_texts[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))


    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]

    def __len__(self):
        return len(self.data)

    def _longest_encoded_length(self):
        max_length = 0
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length
train_dataset = DatasetV1(
    dataset = "tweets_train.csv",
    max_length = 4,
    stride =4,
    tokenizer=tokenizer
)
val_dataset = DatasetV1(
    dataset="tweets_val.csv",
    max_length=train_dataset.max_length,    stride =4,
    tokenizer=tokenizer
)
test_dataset = DatasetV1(
    dataset="tweets_test.csv",
    max_length=train_dataset.max_length,    stride =4,
    tokenizer=tokenizer
)