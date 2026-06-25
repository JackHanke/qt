## NOTE this code is super bad

import os
import torch
from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.decoders import Metaspace as MetaspaceDecoder, Sequence as SequenceDecoder

from qt import qt
from decode import *

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

class Chat:
    def __init__(self):
        # get device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.MAX_TOKENS_ALLOWED_TO_GENERATE = 256

        ## load model with configs
        D_MODEL = 2048
        N_LAYERS = 22
        N_HEADS = 32
        N_HEADS_KV = 8
        SEQ_LEN = 512
        NUM_EMBEDDINGS = 10_001

        self.model = qt(
            d_model=D_MODEL,
            n_layers=N_LAYERS,
            n_heads=N_HEADS,
            n_heads_kv=N_HEADS_KV,
            seq_len=SEQ_LEN,
            num_embeddings=NUM_EMBEDDINGS,
            device=self.device
        ).to(self.device)
        self.model.eval()
        # model_dict = torch.load(model_dir+'checkpoints/qt-pretrain_best.pt')
        MODEL_PATH = Path(f'models/checkpoints/2026-06-20-21:56:34/file_1_posttrain_qt.pth')
        # MODEL_PATH = Path(f'models/checkpoints/2026-06-07-10:05:02/2026-06-07-10:05:02_file_32_pretrain_qt.pth')
        self.model.load_state_dict(torch.load(MODEL_PATH))

        self.logo_str = f'''
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
        self.info_string = '''
        Enter 'r' or 'restart' to restart the conversation.
        Enter 'q' or 'quit' to exit chat window.

        Loading model...'''
        self.context = ''
        self.context_tokens = []

    def _reset_context(self):
        self.context = '[BOS]'
        # self.context = '[BOS][USER]you are qt. you are a billion parameter language model who is helpful and honest.[AI]i am qt.[USER]'
        self.context_tokens = tokenizer.encode(self.context).ids # TODO fix this oh my god
        
    def __call__(self):
        # clear terminal header and print info
        os.system('clear')
        print(self.logo_str)
        print(self.info_string)
    
        # 
        user_prompt_string = f'\n > '
        self._reset_context()
        
        while True:
            user_input = str(input(user_prompt_string))

            if user_input.strip() in ['q', 'quit']:
                print('')
                break
            elif user_input.strip() in ['r', 'restart']:
                self._reset_context()
                print(f'[SYSTEM] Conversation restarted.')
                continue

            user_input =  '[USER]' + preprocess(user_input) + '[AI]'
            # user_input =  preprocess(user_input)
            
            # add user input to context
            self.context += user_input
            self.context_tokens.extend(tokenizer.encode(user_input).ids)
            # print(context_tokens)

            ## model generation
            print(f'{bcolors.CYAN}q{bcolors.ENDC}{bcolors.BLUE}t{bcolors.ENDC}:', end='')

            tokens_generated = 0
            next_token_id = 7
            while next_token_id not in [0, 1, 2, 3, 4, 5] and tokens_generated < self.MAX_TOKENS_ALLOWED_TO_GENERATE:
                with torch.inference_mode():
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        context_tokens_tensor = torch.tensor(self.context_tokens).unsqueeze(0).to(self.device)
                        output_logits = self.model(context_tokens_tensor)

                    logits = output_logits[0, :, -1]

                    # sample
                    next_token_id = top_k_sampling(logits)
                    if next_token_id == 3:
                        # print('[EOS]')
                        next_token_id = 4
                    
                    self.context_tokens.append(next_token_id)
                    tokens_generated += 1

                    # decode
                    next_token = tokenizer.decode([next_token_id])
                    if len(next_token)> 0 and next_token[0] == '▁': next_token = ' ' + next_token[1:] # NOTE war crime
                    # next_token = decoder.decode([temp])
                    print(f"{bcolors.CYAN}{next_token}{bcolors.ENDC}", end='')

                    # DEBUG TODO remove
                    if next_token_id in [0, 1, 2, 5]: print(f'ended with: {next_token_id}')
                    # elif len(next_token) == 0: print(f'error with: {next_token_id}')


if __name__ == '__main__':
    chat = Chat()
    chat()

