import os
import logging
from tqdm import tqdm
from time import time
from math import ceil
from pathlib import Path
from datetime import datetime

import torch
from torchinfo import summary
from torch.utils.data import DataLoader

from models.qt import qt
from data.dataset import PretrainDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# configs
DATA_ROOT = Path(f'data/train/')

LEARNING_RATE = 5e-4
BETA_1 = 0.9
BETA_2 = 0.95
CLIP_NORM = 1.0
LABEL_SMOOTHING = 0.0

### qt 100M
# D_MODEL = 640
# N_LAYERS = 16
# N_HEADS = 10
# EFFECTIVE_BATCH_SIZE = 2_025 # number of sequences, not number of tokens
# TRUE_BATCH_SIZE = 27

### qt ()
D_MODEL = 1792
N_LAYERS = 25
N_HEADS = 14
EFFECTIVE_BATCH_SIZE = 2_025 # number of sequences, not number of tokens
TRUE_BATCH_SIZE = 25

accumulate_every = EFFECTIVE_BATCH_SIZE // TRUE_BATCH_SIZE
SEQ_LEN = 512
NUM_EMBEDDINGS = 10_001

model = qt(
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    seq_len=SEQ_LEN,
    num_embeddings=NUM_EMBEDDINGS,
    device=DEVICE
).to(dtype=torch.bfloat16).to(DEVICE)
model.compile()

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(BETA_1, 0.95), weight_decay=0.1)

total_steps = 20_972

warmup_steps = 1_000
cooldown_steps = 1_000
warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps
)
constant_scheduler = torch.optim.lr_scheduler.ConstantLR(
    optimizer, factor=1.0, total_iters=(total_steps-warmup_steps-cooldown_steps)
)
cooldown_scheduler = torch.optim.lr_scheduler.LinearLR(
    optimizer, start_factor=1.0, end_factor=0.01, total_iters=cooldown_steps
)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    optimizer,
    schedulers=[warmup_scheduler, constant_scheduler, cooldown_scheduler],
    milestones=[warmup_steps, (total_steps-cooldown_steps)]
)

loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING, ignore_index=1) # TODO ignore pad token

total_its, data_to_gpu, forward_passes, backward_passes, opts, prog_bar_times, calc_losses = [], [], [], [], [], [], []

training_files = ['000_00000.parquet']
for file_num, data_path in enumerate(training_files):
    dataset = PretrainDataset(data_path=DATA_ROOT/data_path)
    dataloader = DataLoader(dataset, batch_size=TRUE_BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=True)

    total_start = time()
    tot_its = 4*accumulate_every
    # print(f'tot_its: {tot_its}')

    optimizer.zero_grad()

    total_batches = ceil(len(dataset)/TRUE_BATCH_SIZE)
    prog_bar = tqdm(enumerate(dataloader), total=tot_its)
    for batch_idx, (seq_in, seq_out) in prog_bar:
        total_it_start = time()

        data_to_gpu_start = time()
        seq_in = seq_in.to(DEVICE, non_blocking=True)
        seq_out = seq_out.to(DEVICE, non_blocking=True)
        tim = time()-data_to_gpu_start
        data_to_gpu.append(tim)

        forward_pass_start = time()
        logits = model(seq_in)
        tim = time()-forward_pass_start
        # print(f'{batch_idx} forward pass: {tim}')
        forward_passes.append(tim)

        calc_loss_start = time()
        loss = loss_fn(logits, seq_out)
        # loss_val = loss.item()
        loss = loss / accumulate_every
        calc_losses.append(time()-calc_loss_start)

        backward_pass_start = time()
        loss.backward()
        tim = time()-backward_pass_start
        # print(f'{batch_idx} backward pass: {tim}')
        backward_passes.append(tim)


        if ((batch_idx+1) % accumulate_every) == 0:
            optim_start = time()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_NORM)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # batch_info_str = f'File {file_num+1}/{len(training_files)}, batch {batch_idx+1}/{total_batches} done with train loss: {loss_val:.5f}'
            # logger.info(batch_info_str)

            opts.append(time()-optim_start)
        else:
            # batch_info_str = f'File {file_num+1}/{len(training_files)}, batch {batch_idx+1}/{total_batches} accd with train loss: {loss_val:.5f}'
            # logger.info(batch_info_str)
            pass

        prog_bar_desc_start = time()
        # prog_bar.set_description(batch_info_str)
        prog_bar_times.append(time()-prog_bar_desc_start)


        tim = time()-total_it_start
        # print(f'{batch_idx} total it: {tim}')
        total_its.append(tim)

        if batch_idx == tot_its: break

    tim = time() - total_start
    print(f'whole loop: {tim}')    
    print(f'data iter create: {tim-sum(total_its)}')    
    
print(f'total its: {sum(total_its)/len(total_its)}')
print(f'data  to : {sum(data_to_gpu)/len(data_to_gpu)}')
print(f'fwd   its: {sum(forward_passes)/len(forward_passes)}')
print(f'calc  its: {sum(calc_losses)/len(calc_losses)}')
print(f'bkwd  its: {sum(backward_passes)/len(backward_passes)}')
print(f'opts  its: {sum(opts)/len(opts)}')
print(f'prog  its: {sum(prog_bar_times)/len(prog_bar_times)}')
