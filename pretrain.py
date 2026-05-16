import os
import logging
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from math import ceil

import torch
from torchinfo import summary
from torch.utils.data import DataLoader

from models.qt import qt
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
    DATA_ROOT = Path(f'data/dev/')
    EFFECTIVE_BATCH_SIZE = 1_000 # number of sequences, not number of tokens
    TRUE_BATCH_SIZE = 25
    accumulate_every = EFFECTIVE_BATCH_SIZE // TRUE_BATCH_SIZE

    LEARNING_RATE = 1e-5 # TODO scheduler, warmup and cosine warmdown
    LABEL_SMOOTHING = 0.0

    D_MODEL = 1792
    N_LAYERS = 25
    N_HEADS = 14
    SEQ_LEN = 512
    NUM_EMBEDDINGS = 10_001

    logger.info(f'Starting experiment: {experiment_start_time_str} on device: {DEVICE}')
    logger.info(f'''CONFIGS
    Training Configs:
        EFFECTIVE_BATCH_SIZE:  {EFFECTIVE_BATCH_SIZE}
        TRUE_BATCH_SIZE:       {TRUE_BATCH_SIZE}
        LABEL_SMOOTHING:       {LABEL_SMOOTHING}
    Model Configs:
        D_MODEL:               {D_MODEL}
        N_LAYERS:              {N_LAYERS}
        N_HEADS:               {N_HEADS}
        SEQ_LEN:               {SEQ_LEN}
        NUM_EMBEDDINGS:        {NUM_EMBEDDINGS}
    ''')

    model = qt(
        d_model=D_MODEL,
        ffw_size=4*D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        seq_len=SEQ_LEN,
        num_embeddings=NUM_EMBEDDINGS,
        device=DEVICE
    ).to(DEVICE).to(dtype=torch.bfloat16)
    model_summary_str = str(summary(model))
    logger.info('\n'+model_summary_str)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING, ignore_index=1) # TODO ignore pad token

    training_files = sorted(os.listdir(DATA_ROOT))
    for file_num, data_path in enumerate(training_files):
        dataset = PretrainDataset(data_path=DATA_ROOT/data_path)
        dataloader = DataLoader(dataset, batch_size=TRUE_BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=False)

        total_batches = ceil(len(dataset)/TRUE_BATCH_SIZE)
        prog_bar = tqdm(enumerate(dataloader), total=total_batches)
        for batch_idx, (seq_in, seq_out) in prog_bar:
            seq_in = seq_in.to(DEVICE, non_blocking=True)
            seq_out = seq_out.to(DEVICE, non_blocking=True)

            logits = model(seq_in)

            loss = loss_fn(logits, seq_out)
            loss = loss / accumulate_every
            loss.backward()

            if ((batch_idx+1) % accumulate_every) == 0 or batch_idx+1 == total_batches:
                optimizer.step()
                optimizer.zero_grad()

                batch_info_str = f'File {file_num+1}/{len(training_files)}, batch {batch_idx+1}/{len(dataset)} done with train loss: {loss.item():.5f}'
                logger.info(batch_info_str)
                prog_bar.set_description(batch_info_str)
            else:
                batch_info_str = f'File {file_num+1}/{len(training_files)}, batch {batch_idx+1}/{len(dataset)} accumulated with train loss: {loss.item():.5f}'
                logger.info(batch_info_str)
                prog_bar.set_description(batch_info_str)

        # checkpointing
        checkpoint_path = f'models/checkpoints/{experiment_start_time_str}_file_{file_num}_pretrain_qt.pt'
        torch.save(qt.state_dict(), checkpoint_path)
   




if __name__ == '__main__':
    pretrain()
