import os
import logging
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

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
    BATCH_SIZE = 1_000 # number of sequences, not number of tokens
    ACCUMULATE_BATCH_SIZE = 25
    LEARNING_RATE = 1e-5
    LABEL_SMOOTHING = 0.0

    D_MODEL = 1792
    N_LAYERS = 23
    N_HEADS = 14
    SEQ_LEN = 512
    NUM_EMBEDDINGS = 13_000

    logger.info(f'Starting experiment: {experiment_start_time_str} on device: {DEVICE}')
    logger.info(f'''CONFIGS
    Training Configs:
        BATCH_SIZE:            {BATCH_SIZE}
        ACCUMULATE_BATCH_SIZE: {ACCUMULATE_BATCH_SIZE}
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

    train_loss = 0
    training_files = os.listdir(DATA_ROOT)
    for file_num, data_path in enumerate(training_files):
        dataset = PretrainDataset(data_path=DATA_ROOT/data_path)
        dataloader = DataLoader(dataset, batch_size=ACCUMULATE_BATCH_SIZE, shuffle=False, pin_memory=True)

        total_batches = (len(dataset)//ACCUMULATE_BATCH_SIZE)+1
        prog_bar = tqdm(enumerate(dataloader), total=total_batches)
        for batch_idx, (seq_in, seq_out) in prog_bar:
            seq_in = seq_in.to(DEVICE, non_blocking=True)
            seq_out = seq_out.to(DEVICE, non_blocking=True)

            logits = model(seq_in)

            loss = loss_fn(logits, seq_out)
            train_loss += loss.item() # TODO fix so the value doesnt below

            loss.backward()
            if ((batch_idx+1) % (BATCH_SIZE//ACCUMULATE_BATCH_SIZE)) == 0 or batch_idx+1 == total_batches:
                optimizer.step()
                optimizer.zero_grad()

            batch_info_str = f'File {file_num+1}/{len(training_files)} ({data_path}), batch {batch_idx} completed with train loss: {loss.item():.5f}'
            logger.info(batch_info_str)
            prog_bar.set_description(batch_info_str)

        # TODO checkpointing




if __name__ == '__main__':
    pretrain()
