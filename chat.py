import os
import torch
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder, Sequence as SequenceDecoder

from qt import qt

MAX_TOKENS_ALLOWED_TO_GENERATE = 256
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

'''

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
model.inference_mode()
# model_dict = torch.load(model_dir+'checkpoints/qt-pretrain_best.pt')
MODEL_PATH = Path(f'/home/jack/vault/software/qt/models/checkpoints/2026-06-07-10:05:02/2026-06-07-10:05:02_file_32_pretrain_qt.pth')
model.load_state_dict(torch.load(MODEL_PATH))

# 
user_prompt_string = f'  Talk to {bcolors.CYAN}q{bcolors.ENDC}{bcolors.BLUE}t{bcolors.ENDC}: '

context = '[BOS_TOKEN]'
while True:
    user_input = str(input(user_prompt_string))

    if user_input in ['q', 'quit']:
        print('')
        break

    user_input =  '[USER]' + preprocess(user_input) + '[AI]'
    
    # add user input to context
    context += user_input

    context_tokens = tokenizer.encode(context).ids

    
    tokens_generated = 0
    while next_token not in ['[EOS]', '[USER]', '[AI]'] and tokens_generated < MAX_TOKENS_ALLOWED_TO_GENERATE:
        with torch.inference_mode():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                context_tokens_tensor = torch.tensor(context_tokens).unsqueeze(0).to(DEVICE)
                output_logits = model(context_tokens_tensor)

            # decode
            probs = torch.nn.functional.softmax(output_logits[0, :, -1])

            next_token_id = None

            next_token = decoder.decode(next_token_id)
            tokens_generated += 1
            print(f"{bcolors.CYAN}{next_token:>12}{bcolors.ENDC}", end='')

    print(f"{bcolors.CYAN}{model_response:>12}{bcolors.ENDC}")

    # add response to context
    context += model_response



