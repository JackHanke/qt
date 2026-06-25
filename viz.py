## note this is modified starter code from Gemini

import arcade
import random
import os
import numpy as np
import torch
from pathlib import Path
from sklearn.decomposition import PCA

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

pca = PCA(n_components=2)
pcaed_embeds = pca.fit_transform(model.embeddings.weight.clone().detach().cpu().numpy())

print('Model loaded')

SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
SCREEN_TITLE = "Typing Animation Example"

class TypingAnimationApp(arcade.Window):
    def __init__(self):
        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)
        
        # Application state variables
        self.background_color = arcade.color.WHITE
        self.current_text = ""
        self.context_tokens = []
        self.token_points = []
        self.current_points = []
        self.activations = None
        self.activation_idx = 1
        self.activation_interp_count = 0
        self.activation_interp_total = 10
        self.generated_token = None
        self.generation_done = False

        # Animation variables
        self.circle_radius = 6
        self.circle_color = arcade.color.BLACK

    def on_draw(self):
        """ Render the screen. """
        self.clear()

        ## AUTO SCALE AND CENTER
        if len(self.current_points) >= 2:
            min_x = min([self.current_points[i][0] for i in range(len(self.current_points))])
            max_x = max([self.current_points[i][0] for i in range(len(self.current_points))])
            min_y = min([self.current_points[i][1] for i in range(len(self.current_points))])
            max_y = max([self.current_points[i][1] for i in range(len(self.current_points))])
            # print(f'min_x: {min_x:.3f} max_x: {max_x:.3f} min_y: {min_y:.3f} max_y: {max_y:.3f}')

            global_shrinkage = 0.55
            x_scale = SCREEN_WIDTH/abs(max_x-min_x)*global_shrinkage
            y_scale = SCREEN_HEIGHT/abs(max_y-min_y)*global_shrinkage

            mid_x = (SCREEN_WIDTH/2) - x_scale*(max_x+min_x)/2
            mid_y = (SCREEN_HEIGHT/2) - y_scale*(max_y+min_y)/2

            # Draw the animated circle if it has a radius
            for point_idx in range(len(self.current_points)):
                if self.activations is not None or self.generation_done: color = arcade.color.RED
                else: color = self.circle_color
                if point_idx != len(self.current_points)-1:
                    arcade.draw_line(
                        x_scale*self.current_points[point_idx][0]+mid_x,
                        y_scale*self.current_points[point_idx][1]+mid_y,
                        x_scale*self.current_points[point_idx+1][0]+mid_x,
                        y_scale*self.current_points[point_idx+1][1]+mid_y,
                        color=color,
                        line_width=3,
                    )
                arcade.draw_circle_filled(
                    x_scale*self.current_points[point_idx][0]+mid_x,
                    y_scale*self.current_points[point_idx][1]+mid_y,
                    self.circle_radius,
                    color
                )
        
        if self.generation_done:
            arcade.draw_text(
                f"{self.generated_token}",
                x=x_scale*self.current_points[point_idx][0]+mid_x + 10,
                y=y_scale*self.current_points[point_idx][1]+mid_y - 10,
                width=700,
                color=color,
                font_name='Liberation Mono',
                font_size=12,
                bold=True,
                multiline=True
            )
            

        # Draw the text the user has typed
        arcade.draw_text(
            f"{self.current_text}",
            x=50,
            y=75,
            width=700,
            color=self.circle_color,
            font_name='Liberation Mono',
            font_size=12,
            bold=True,
            multiline=True
        )

    def on_update(self, delta_time):
        """ Movement and animation logic (runs ~60 times a second) """

        if self.activations is not None:
            # print(f'len(acts): {len(self.activations)}')

            self.activation_interp_count += 1
            if self.activation_interp_count >= self.activation_interp_total:
                self.activation_interp_count = 0
                self.activation_idx += 1
                # print(f'act idx: {self.activation_idx}')
                if self.activation_idx == len(self.activations) - 1:
                    self.activations  = None
                    self.activation_idx = 1
                    self.generation_done = True
                    return

            for i in range(2, len(self.current_points)):
                x_2, y_2 = self.activations[self.activation_idx][i] # skip bos and user token
                x_c, y_c = self.current_points[i]

                # if self.activation_idx > 2:
                #     x_0, y_0 = self.activations[self.activation_idx-2][i]
                #     x_1, y_1 = self.activations[self.activation_idx-1][i]

                #     x_3, y_3 = (x_0+x_1, y_0+y_1)

                #     target_x = x_3 + (x_2 - x_3)*(self.activation_interp_count/self.activation_interp_total)
                #     target_y = y_3 + (y_2 - y_3)*(self.activation_interp_count/self.activation_interp_total)

                #     self.current_points[i][0] = x_c + (target_x - x_c)*(self.activation_interp_count/self.activation_interp_total)
                #     self.current_points[i][1] = y_c + (target_y - y_c)*(self.activation_interp_count/self.activation_interp_total)
                # else:
                self.current_points[i][0] = x_c + (x_2 - x_c)/self.activation_interp_total
                self.current_points[i][1] = y_c + (y_2 - y_c)/self.activation_interp_total

    def on_key_press(self, key, modifiers):
        """ Triggered instantly every time a key is pressed """
        
        # Handle Backspace
        if key == arcade.key.BACKSPACE:
            self.current_text = self.current_text[:-1]
            self.context_tokens = tokenizer.encode(self.current_text).ids
            self.token_points = pcaed_embeds[self.context_tokens]
            self.current_points = pcaed_embeds[self.context_tokens]
            return
        
        # Handle Enter
        if key == arcade.key.ENTER:
            with torch.inference_mode():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    context_tokens_tensor = torch.tensor([2,4]+self.context_tokens+[5]).unsqueeze(0).to(DEVICE)
                    output_logits, activations = model(context_tokens_tensor, do_viz=True)

                    temp = []
                    for act in activations:
                        temp.append(pca.transform(act.squeeze(0).numpy()))
                    self.activations = temp

                # sample
                # next_token_id = top_k_sampling(output_logits[0:1, :, -1]).item()
                # print(f'next token id: {next_token_id}')
                preds = torch.argmax(output_logits, dim=1).squeeze(0)
                next_token_id = preds[-1].item()
                next_token = tokenizer.decode([next_token_id])
                self.generated_token = next_token[1:]
                print(f'next_token: {next_token}')

            return
            
        # Convert the key code to an actual character
        char = chr(key) if 32 <= key <= 126 else ""
        
        if char:
            self.current_text += char
            self.context_tokens = tokenizer.encode(self.current_text).ids
            self.token_points = pcaed_embeds[self.context_tokens]
            self.current_points = pcaed_embeds[self.context_tokens]


def main():
    app = TypingAnimationApp()
    arcade.run()

if __name__ == "__main__":
    main()