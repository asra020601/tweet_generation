import torch
from data import tokenizer
prompt = "Hello world"
input_ids = tokenizer.encode(prompt, return_tensors='pt')  # shape: [1, seq_len]
def generate_text_simple(model, input_ids, max_new_tokens=20, temperature=1.0):
    """
    Generate text using a simple sampling approach.
    
    Args:
        model: The transformer model
        input_ids (torch.Tensor): Input token ids of shape [seq_len] or [batch_size, seq_len]
        max_new_tokens (int): Maximum number of tokens to generate
        temperature (float): Sampling temperature
    
    Returns:
        torch.Tensor: Generated token ids
    """
    # Add batch dimension if it's missing
    if len(input_ids.shape) == 1:
        input_ids = input_ids.unsqueeze(0)  # Shape: [1, seq_len]
    
    # Generate tokens one at a time
    for _ in range(max_new_tokens):
        # Perform a forward pass
        outputs = model(input_ids)
        
        # Extract logits from outputs
        logits = outputs[0] if isinstance(outputs, tuple) else outputs.logits
        
        # Get logits for the last token
        logits = logits[:, -1, :]  # Shape: [batch_size, vocab_size]
        
        # Apply temperature
        logits = logits / temperature
        
        # Convert to probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Sample next token
        next_token = torch.multinomial(probs, num_samples=1)  # Shape: [batch_size, 1]
        
        # Append the new token
        input_ids = torch.cat([input_ids, next_token], dim=-1)
    
    # Remove batch dimension if it wasn't originally present
    if len(input_ids.shape) == 2 and input_ids.shape[0] == 1:
        input_ids = input_ids.squeeze(0)
        
    return input_ids

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)  # remove batch dimension
    return tokenizer.decode(flat.tolist())
def calc_loss_batch(input_batch, target_batch, model):
    logits = model(input_batch).logits  # Access the logits directly from the model output
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target_batch.flatten())  # Flatten the logits and target
    return loss


def calc_loss_loader(data_loader, model, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches
def evaluate_model(model, train_loader, val_loader, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, start_context="hello"):
    model.eval()
    context_size = model.transformer.wpe.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer)
    with torch.no_grad():
        token_ids = generate_text_simple(model, input_ids, max_new_tokens=20, temperature=1.0)
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()
