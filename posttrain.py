import os
import logging
import numpy as np
import random
from tqdm import tqdm
from math import ceil
from pathlib import Path
from datetime import datetime

import torch
from torchinfo import summary
from torch.utils.data import DataLoader

from qt import qt
from data.dataset import PretrainDataset


def posttrain():
    experiment_start_time = datetime.now()
    experiment_start_time_str = experiment_start_time.strftime("%Y-%m-%d-%H:%M:%S")

    checkpoint_path = Path(f'./models/checkpoints/{experiment_start_time_str}')
    if not os.path.exists(checkpoint_path): os.makedirs(checkpoint_path)

    logger = logging.getLogger(__name__)
    # logger.addHandler(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        filename=f'logs/posttraining-{experiment_start_time_str}.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # configs
    DATA_ROOT = Path(f'data/posttraining/')

    MODEL_PATH = Path(f'models/checkpoints/2026-06-07-10:05:02/2026-06-07-10:05:02_file_32_pretrain_qt.pth')

    TRUE_BATCH_SIZE = 8
    accumulate_every = ceil(128/TRUE_BATCH_SIZE)
    EFFECTIVE_BATCH_SIZE = accumulate_every*TRUE_BATCH_SIZE

    LEARNING_RATE = 1e-5
    BETA_1 = 0.9
    BETA_2 = 0.95
    WEIGHT_DECAY = 0.1
    CLIP_NORM = 1.0
    LABEL_SMOOTHING = 0.0

    ### qt config
    D_MODEL = 2048
    N_LAYERS = 22
    N_HEADS = 32
    N_HEADS_KV = 8 

    INIT_MEAN = 0.0
    INIT_STD = 0.02

    SEQ_LEN = 1024
    NUM_EMBEDDINGS = 10_001

    SEED = 4
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    configs_str = f'''Starting experiment: {experiment_start_time_str} on device: {DEVICE}
    CONFIGS
    Training Configs:
        SEED:                  {SEED}
        EFFECTIVE_BATCH_SIZE:  {EFFECTIVE_BATCH_SIZE}
        TRUE_BATCH_SIZE:       {TRUE_BATCH_SIZE}
        accumulate_every:      {accumulate_every}
        LABEL_SMOOTHING:       {LABEL_SMOOTHING}
        BETA_1:                {BETA_1}
        BETA_2:                {BETA_2}
        WEIGHT_DECAY:          {WEIGHT_DECAY}
        CLIP_NORM:             {CLIP_NORM}
    Model Configs:
        D_MODEL:               {D_MODEL}
        N_LAYERS:              {N_LAYERS}
        N_HEADS:               {N_HEADS}
        N_HEADS_KV:            {N_HEADS_KV}
        SEQ_LEN:               {SEQ_LEN}
        NUM_EMBEDDINGS:        {NUM_EMBEDDINGS}
        INIT_MEAN:             {INIT_MEAN}
        INIT_STD:              {INIT_STD}
    '''
    logger.info(configs_str)

    model = qt(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        n_heads_kv=N_HEADS_KV,
        seq_len=SEQ_LEN,
        num_embeddings=NUM_EMBEDDINGS,
        device=DEVICE
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    print(f'Model weights loaded from: {MODEL_PATH}')
    model.compile()

    model_summary_str = str(summary(model))
    logger.info('\n'+model_summary_str)

    print(configs_str)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2), weight_decay=WEIGHT_DECAY)

    scheduler = None
    
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING, ignore_index=1) # TODO ignore pad token

    training_files = [Path('data.parquet')]
    for file_num, data_path in enumerate(training_files):
        dataset = PretrainDataset(data_path=DATA_ROOT/data_path)
        dataloader = DataLoader(dataset, batch_size=TRUE_BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=True)

        optimizer.zero_grad()

        loss_batch_val, loss_batch_val_temp = 0, 0

        total_iters_per_file = (len(dataset) // EFFECTIVE_BATCH_SIZE)*accumulate_every
        prog_bar = tqdm(enumerate(dataloader), total=total_iters_per_file)
        for batch_idx, (seq_in, seq_out) in prog_bar:
            seq_in = seq_in.to(DEVICE, non_blocking=True)
            seq_out = seq_out.to(DEVICE, non_blocking=True)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(seq_in)

                loss = loss_fn(logits, seq_out)
                loss_val = loss.item()
                loss_batch_val_temp += loss_val / accumulate_every
                loss = loss / accumulate_every

            loss.backward()

            if ((batch_idx+1) % accumulate_every) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_NORM)

                optimizer.step()
                if scheduler is not None: scheduler.step()
                optimizer.zero_grad()

                loss_batch_val = loss_batch_val_temp
                loss_batch_val_temp = 0

                batch_info_str = f'File {file_num+1}/{len(training_files)}, done train loss iter: {loss_val:.5f} batch: {loss_batch_val:.5f}'
                logger.info(batch_info_str)
                prog_bar.set_description(batch_info_str)

            else:
                batch_info_str = f'File {file_num+1}/{len(training_files)}, accd train loss iter: {loss_val:.5f} batch: {loss_batch_val:.5f}'
                logger.info(batch_info_str)
                prog_bar.set_description(batch_info_str)

            # break out early for batch rounding
            if batch_idx == (total_iters_per_file - 1): break

        # checkpointing
        checkpoint_path = f'models/checkpoints/{experiment_start_time_str}/file_{file_num+1}_posttrain_qt.pth'
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f'Checkpointed at: {checkpoint_path}')
   

if __name__ == '__main__':

    posttrain()
