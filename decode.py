import torch

def greedy_sampling(logits):
    next_token_id = torch.argmax(logits, dim=0).squeeze(0)
    return next_token_id.item()

def top_k_sampling(logits, k=10, temperature=1.0):
    # 1. Apply temperature scaling to logits
    logits = logits / temperature
    # 2. Get the top-k logits and their indices
    top_k_logits, top_k_indices = torch.topk(logits, k)
    # 3. Filter out other logits by setting them to -infinity
    # Create a mask for tokens not in top-k
    filtered_logits = logits.new_ones(logits.shape) * float('-inf')
    # Scatter the top-k values back into the filtered tensor
    filtered_logits.scatter_(0, top_k_indices, top_k_logits)
    # 4. Convert filtered logits to probabilities
    probabilities = torch.nn.functional.softmax(filtered_logits, dim=0)
    # 5. Sample the next token from the top-k distribution
    next_token = torch.multinomial(probabilities, num_samples=1)
    return next_token.item()

def top_p_sampling(logits, p=0.9, temperature=1.0):
    logits = logits / temperature


def beam_search():
    pass
