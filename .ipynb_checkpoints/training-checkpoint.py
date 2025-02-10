
from transformers import AutoTokenizer, GPT2LMHeadModel
import torch
import time
import pandas as pd
from torch.utils.data import DataLoader
from dataloader import train_loader, val_loader,val_dataset
from data import val_data
from evaluate import calc_loss_batch, evaluate_model, generate_and_print_sample
from transformers import AutoModel
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false" 
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

def train_model_simple(model, train_loader, val_loader, optimizer, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        generate_and_print_sample(
            model, tokenizer, start_context
        )

    return train_losses, val_losses, track_tokens_seen

start_time = time.time()


optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

num_epochs = 10

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context=val_data.iloc[0]["text"], tokenizer=tokenizer
)

end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")
model.save_pretrained("saved_model")
torch.save({'epoch': epoch, 'model_state_dict': model.state_dict()}, "checkpoint.pth")
