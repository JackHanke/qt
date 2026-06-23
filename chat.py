## NOTE this code is super bad

import os
import torch
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder, Sequence as SequenceDecoder

from qt import qt

MAX_TOKENS_ALLOWED_TO_GENERATE = 128
tokenizer = Tokenizer.from_file("data/tokenizer.json")
decoder = SequenceDecoder([MetaspaceDecoder()])

def preprocess(text: str):
    punctuation_map = {
        0x201C: 0x22,  # LEFT DOUBLE QUOTATION MARK -> "
        0x201D: 0x22,  # RIGHT DOUBLE QUOTATION MARK -> "
        0x2018: 0x27,  # LEFT SINGLE QUOTATION MARK -> '
        0x2019: 0x27,  # RIGHT SINGLE QUOTATION MARK -> '
    }
    return text.translate(punctuation_map).lower().replace(r"[^a-z0-9 [:space:][:punct:]]", '')

def top_k_sampling(logits, k=5, temperature=1.0):
    # 1. Apply temperature scaling to logits
    logits = logits / temperature
    # 2. Get the top-k logits and their indices
    top_k_logits, top_k_indices = torch.topk(logits, k)
    # 3. Filter out other logits by setting them to -infinity
    # Create a mask for tokens not in top-k
    filtered_logits = logits.new_ones(logits.shape) * float('-inf')
    # Scatter the top-k values back into the filtered tensor
    filtered_logits.scatter_(1, top_k_indices, top_k_logits)
    # 4. Convert filtered logits to probabilities
    probabilities = torch.nn.functional.softmax(filtered_logits, dim=-1)
    # 5. Sample the next token from the top-k distribution
    next_token = torch.multinomial(probabilities, num_samples=1)
    return next_token

class bcolors:
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    ENDC = '\033[0m'

logo_str = f'''
{bcolors.BOLD}{bcolors.CYAN}           ___    {bcolors.ENDC}{bcolors.BLUE}    ___             {bcolors.ENDC}
{bcolors.BOLD}{bcolors.CYAN}          /\  \   {bcolors.ENDC}{bcolors.BLUE}   /\  \            {bcolors.ENDC}
{bcolors.BOLD}{bcolors.CYAN}         /88\  \  {bcolors.ENDC}{bcolors.BLUE}   \8\  \           {bcolors.ENDC}
{bcolors.BOLD}{bcolors.CYAN}        /8/\8\  \ {bcolors.ENDC}{bcolors.BLUE}    \8\  \          {bcolors.ENDC}
{bcolors.BOLD}{bcolors.CYAN}        \8\~\8\__\{bcolors.ENDC}{bcolors.BLUE}    /88\  \         {bcolors.ENDC}
{bcolors.BOLD}{bcolors.CYAN}         \8\/8/  /{bcolors.ENDC}{bcolors.BLUE}   /8/\8\__\        {bcolors.ENDC}
{bcolors.BOLD}{bcolors.CYAN}          \88/  / {bcolors.ENDC}{bcolors.BLUE}  /8/  \/__/        {bcolors.ENDC}
{bcolors.BOLD}{bcolors.CYAN}          /8/__/  {bcolors.ENDC}{bcolors.BLUE} /8/  /             {bcolors.ENDC}
{bcolors.BOLD}{bcolors.CYAN}          \8\__\  {bcolors.ENDC}{bcolors.BLUE} \/__/              {bcolors.ENDC}
{bcolors.BOLD}{bcolors.CYAN}           \/__/  {bcolors.ENDC}{bcolors.BLUE}                    {bcolors.ENDC}                                   
'''

info_string = '''
Enter 'q' or 'quit' to exit chat window.

Loading model...'''
# TODO add flush context command

# clear terminal header and print info
os.system('clear')
print(logo_str)
print(info_string)

# get device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## load model with configs
D_MODEL = 2048
N_LAYERS = 22
N_HEADS = 32
N_HEADS_KV = 8
SEQ_LEN = 512
NUM_EMBEDDINGS = 10_001

model = qt(
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    n_heads_kv=N_HEADS_KV,
    seq_len=SEQ_LEN,
    num_embeddings=NUM_EMBEDDINGS,
    device=DEVICE
).to(DEVICE)
model.eval()
# model_dict = torch.load(model_dir+'checkpoints/qt-pretrain_best.pt')
MODEL_PATH = Path(f'models/checkpoints/2026-06-20-21:56:34/file_1_posttrain_qt.pth')
# MODEL_PATH = Path(f'models/checkpoints/2026-06-07-10:05:02/2026-06-07-10:05:02_file_32_pretrain_qt.pth')
model.load_state_dict(torch.load(MODEL_PATH))


def chat():
    # 
    user_prompt_string = f'\n## Talk to {bcolors.CYAN}q{bcolors.ENDC}{bcolors.BLUE}t{bcolors.ENDC}: '

    context = '[BOS][USER]'
    # context = '[BOS][USER]you are qt. you are a billion parameter language model who is helpful and honest.'
    context_tokens = [2,4] # TODO fix this oh my god
    while True:
        user_input = str(input(user_prompt_string))

        if user_input in ['q', 'quit']:
            print('')
            break

        user_input =  preprocess(user_input) + '[AI]'
        # user_input =  preprocess(user_input)
        
        # add user input to context
        context += user_input
        context_tokens.extend(tokenizer.encode(user_input).ids)
        # print(context_tokens)

        ## model generation
        tokens_generated = 0
        next_token_id = 7
        while next_token_id not in [0, 1, 2, 3, 4, 5] and tokens_generated < MAX_TOKENS_ALLOWED_TO_GENERATE:
            with torch.inference_mode():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    context_tokens_tensor = torch.tensor(context_tokens).unsqueeze(0).to(DEVICE)
                    output_logits = model(context_tokens_tensor)

                # sample
                # next_token_id = top_k_sampling(output_logits[0:1, :, -1]).item()
                # print(f'next token id: {next_token_id}')
                preds = torch.argmax(output_logits, dim=1).squeeze(0)
                next_token_id = preds[-1].item()
                context_tokens.append(next_token_id)
                tokens_generated += 1

                # decode
                next_token = tokenizer.decode([next_token_id])
                if len(next_token)> 0 and next_token[0] == '▁': next_token = ' ' + next_token[1:] # NOTE war crime
                # next_token = decoder.decode([temp])
                print(f"{bcolors.CYAN}{next_token}{bcolors.ENDC}", end='')
                if next_token_id in [0, 1, 2, 3, 4, 5]: print(f'ended with: {next_token_id}')
                elif len(next_token) == 0: print(f'error with: {next_token_id}')


    # if tokens_generated >= MAX_TOKENS_ALLOWED_TO_GENERATE:


    # print(f'after gen: {context_tokens}')

    # print(f"{bcolors.CYAN}{model_response:>12}{bcolors.ENDC}")

    # add response to context
    # context += model_response


if __name__ == '__main__':
    pass

