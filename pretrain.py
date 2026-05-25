import os
import logging
from tqdm import tqdm
from math import ceil
from pathlib import Path
from datetime import datetime

import torch
from torchinfo import summary
from torch.utils.data import DataLoader

from qt import qt
from data.dataset import PretrainDataset


def pretrain():
    experiment_start_time = datetime.now()
    experiment_start_time_str = experiment_start_time.strftime("%Y-%m-%d-%H:%M:%S")

    logger = logging.getLogger(__name__)
    # logger.addHandler(logging.StreamHandler(sys.stdout))
    logging.basicConfig(
        filename=f'logs/pretraining-{experiment_start_time_str}.log',
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # configs
    DATA_ROOT = Path(f'data/train/')
    TRUE_BATCH_SIZE = 30
    accumulate_every = ceil(2_000/TRUE_BATCH_SIZE)
    EFFECTIVE_BATCH_SIZE = accumulate_every*TRUE_BATCH_SIZE

    LEARNING_RATE = 2.5e-4
    BETA_1 = 0.9
    BETA_2 = 0.95
    WEIGHT_DECAY = 0.1
    CLIP_NORM = 1.0
    LABEL_SMOOTHING = 0.0

    ### qt config
    D_MODEL = 2048
    N_LAYERS = 1
    N_HEADS = 32
    N_HEADS_KV = 8 

    SEQ_LEN = 512
    NUM_EMBEDDINGS = 10_001

    warmup_steps = 1_000
    cooldown_steps = 2_000

    configs_str = f'''Starting experiment: {experiment_start_time_str} on device: {DEVICE}
    CONFIGS
    Training Configs:
        EFFECTIVE_BATCH_SIZE:  {EFFECTIVE_BATCH_SIZE}
        TRUE_BATCH_SIZE:       {TRUE_BATCH_SIZE}
        accumulate_every:      {accumulate_every}
        LABEL_SMOOTHING:       {LABEL_SMOOTHING}
        BETA_1:                {BETA_1}
        BETA_2:                {BETA_2}
        WEIGHT_DECAY:          {WEIGHT_DECAY}
        CLIP_NORM:             {CLIP_NORM}
        WSD Warmup Steps:      {warmup_steps}
        WSD Cooldown Steps:    {cooldown_steps}
    Model Configs:
        D_MODEL:               {D_MODEL}
        N_LAYERS:              {N_LAYERS}
        N_HEADS:               {N_HEADS}
        N_HEADS_KV:            {N_HEADS_KV}
        SEQ_LEN:               {SEQ_LEN}
        NUM_EMBEDDINGS:        {NUM_EMBEDDINGS}
    '''
    print(configs_str)
    logger.info(configs_str)

    model = qt(
        d_model=D_MODEL,
        ffw_size=4*D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        n_heads_kv=N_HEADS_KV,
        seq_len=SEQ_LEN,
        num_embeddings=NUM_EMBEDDINGS,
        device=DEVICE
    ).to(dtype=torch.bfloat16).to(DEVICE)
    model.compile()

    model_summary_str = str(summary(model))
    logger.info('\n'+model_summary_str)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2), weight_decay=WEIGHT_DECAY)

    total_steps = 20_854
    
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

    training_files = sorted(os.listdir(DATA_ROOT))
    for file_num, data_path in enumerate(training_files):
        dataset = PretrainDataset(data_path=DATA_ROOT/data_path)
        dataloader = DataLoader(dataset, batch_size=TRUE_BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=True)

        optimizer.zero_grad()

        total_batches = len(dataset)//TRUE_BATCH_SIZE
        prog_bar = tqdm(enumerate(dataloader), total=total_batches)
        for batch_idx, (seq_in, seq_out) in prog_bar:
            seq_in = seq_in.to(DEVICE, non_blocking=True)
            seq_out = seq_out.to(DEVICE, non_blocking=True)

            logits = model(seq_in)

            loss = loss_fn(logits, seq_out)
            loss_val = loss.item()
            loss = loss / accumulate_every
            loss.backward()

            if ((batch_idx+1) % accumulate_every) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_NORM)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                batch_info_str = f'File {file_num+1}/{len(training_files)}, batch {batch_idx+1}/{len(dataset)} done with train loss: {loss_val:.5f}'
                logger.info(batch_info_str)
                prog_bar.set_description(batch_info_str)

                # break out early for batch rounding
                if batch_idx == ((len(dataset) // EFFECTIVE_BATCH_SIZE)*accumulate_every - 1): break
            else:
                batch_info_str = f'File {file_num+1}/{len(training_files)}, batch {batch_idx+1}/{len(dataset)} accd with train loss: {loss_val:.5f}'
                logger.info(batch_info_str)
                prog_bar.set_description(batch_info_str)

        # checkpointing
        checkpoint_path = f'models/checkpoints/{experiment_start_time_str}_file_{file_num}_pretrain_qt.pth'
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f'Checkpointed at: {checkpoint_path}')
   

if __name__ == '__main__':

    pretrain()
