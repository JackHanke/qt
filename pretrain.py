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

# from qt import qt
from qt2 import qtConfig, qt2
from data.dataset import PretrainDataset

# TODO token superposition training? (https://nousresearch.com/token-superposition)
# TODO no decay on embeddings

def get_scheduler(
        optimizer,
        total_steps,
        warmup_steps,
        cooldown_steps,
    ):
    start_factor = 0.01
    end_factor = 0.01
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=start_factor, end_factor=1.0, total_iters=warmup_steps
    )
    constant_scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=1.0, total_iters=(total_steps-warmup_steps-cooldown_steps)
    )
    cooldown_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=end_factor, total_iters=cooldown_steps
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, constant_scheduler, cooldown_scheduler],
        milestones=[warmup_steps, (total_steps-cooldown_steps)]
    )
    return scheduler


def pretrain():
    experiment_start_time = datetime.now()
    experiment_start_time_str = experiment_start_time.strftime("%Y-%m-%d-%H:%M:%S")

    checkpoint_path = Path(f'./models/checkpoints/{experiment_start_time_str}')
    if not os.path.exists(checkpoint_path): os.makedirs(checkpoint_path)

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
    DATA_ROOT = Path(f'data/pretraining/train/')

    MODEL_PATH = None
    # MODEL_PATH = Path(f'models/checkpoints/2026-05-27-20:56:31_file_21_pretrain_qt.pth')

    SEED = 4
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # training
    total_steps = 20_854 # TODO this should not be hardcoded
    TRUE_BATCH_SIZE = 2
    accumulate_every = ceil(2_000/TRUE_BATCH_SIZE)
    EFFECTIVE_BATCH_SIZE = accumulate_every*TRUE_BATCH_SIZE
    SEQ_LEN = 512
    # loss
    LABEL_SMOOTHING = 0.0
    # optimizer
    LEARNING_RATE = 2.5e-4
    BETA_1 = 0.9
    BETA_2 = 0.95
    WEIGHT_DECAY = 0.1
    CLIP_NORM = 1.0
    # scheduler
    WARMUP_STEPS = 1_000
    COOLDOWN_STEPS = 2_000


    configs_str = f'''Starting experiment: {experiment_start_time_str} on device: {DEVICE}
    CONFIGS
    Training Loop Configs:
        total_steps:           {total_steps}
        SEED:                  {SEED}
        EFFECTIVE_BATCH_SIZE:  {EFFECTIVE_BATCH_SIZE}
        TRUE_BATCH_SIZE:       {TRUE_BATCH_SIZE}
        accumulate_every:      {accumulate_every}
        SEQ_LEN:               {SEQ_LEN}
    Loss Configs:
        LABEL_SMOOTHING:       {LABEL_SMOOTHING}
    Optimizer Configs:
        BETA_1:                {BETA_1}
        BETA_2:                {BETA_2}
        WEIGHT_DECAY:          {WEIGHT_DECAY}
        CLIP_NORM:             {CLIP_NORM}
    Scheduler Configs:
        WSD Warmup Steps:      {WARMUP_STEPS}
        WSD Cooldown Steps:    {COOLDOWN_STEPS}
    Model Configs:
        D_MODEL:               {qtConfig.D_MODEL}
        N_LAYERS:              {qtConfig.N_LAYERS}
        N_HEADS:               {qtConfig.N_HEADS}
        N_HEADS_KV:            {qtConfig.N_HEADS_KV}
        NUM_EMBEDDINGS:        {qtConfig.NUM_EMBEDDINGS}
        INIT_MEAN:             {qtConfig.INIT_MEAN}
        INIT_STD:              {qtConfig.INIT_STD}
    '''
    logger.info(configs_str)

    model = qt2(
        d_model=qtConfig.D_MODEL,
        n_layers=qtConfig.N_LAYERS,
        n_heads=qtConfig.N_HEADS,
        n_heads_kv=qtConfig.N_HEADS_KV,
        num_embeddings=qtConfig.NUM_EMBEDDINGS,
        seq_len=SEQ_LEN,
        device=DEVICE
    ).to(DEVICE)

    # TODO fix init, is this not happening to every weight? also clip!
    if MODEL_PATH is None:
        def init_weights(m):
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(m.weight, mean=qtConfig.INIT_MEAN, std=qtConfig.INIT_STD)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        model.apply(init_weights)
    else:
        model.load_state_dict(torch.load(MODEL_PATH))
        print(f'Model weights loaded from: {MODEL_PATH}')

    model.compile()

    model_summary_str = str(summary(model))
    logger.info('\n'+model_summary_str)

    print(configs_str)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(BETA_1, BETA_2), weight_decay=WEIGHT_DECAY)
    
    scheduler = get_scheduler(
        optimizer=optimizer,
        total_steps=total_steps,
        warmup_steps=WARMUP_STEPS,
        cooldown_steps=COOLDOWN_STEPS,
    )
    
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING, ignore_index=1) # TODO ignore pad token

    training_files = sorted(os.listdir(DATA_ROOT))
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
                scheduler.step()
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
            
            if batch_idx == (total_iters_per_file - 1)//2:
                checkpoint_path = f'models/checkpoints/{experiment_start_time_str}_file_{file_num+1}_half_pretrain_qt.pth'
                torch.save(model.state_dict(), checkpoint_path)
                logger.info(f'Checkpointed at: {checkpoint_path}')

            # break out early for batch rounding
            if batch_idx == (total_iters_per_file - 1): break

        # checkpointing
        checkpoint_path = f'models/checkpoints/{experiment_start_time_str}_file_{file_num+1}_pretrain_qt.pth'
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f'Checkpointed at: {checkpoint_path}')
   

if __name__ == '__main__':

    pretrain()
