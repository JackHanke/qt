import os
import logging
from tqdm import tqdm
from pathlib import Path
from datetime import datetime

import torch
from torchsummary import summary
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
    BATCH_SIZE = 500_000
    ACCUMULATE_BATCH_SIZE = 10
    LEARNING_RATE = 1e-5
    LABEL_SMOOTHING = 0.0

    D_MODEL = 128
    N_LAYERS = 2
    N_HEADS = 8
    SEQ_LEN = 512

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
    ''')

    model = qt(
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_heads=N_HEADS,
        seq_len=SEQ_LEN,
        num_embeddings=2,
        device=DEVICE
    ).to(DEVICE)
    model_summary_str = summary(model) + '\n'
    logger.info(model_summary_str)

    optimizer = torch.optim.Adam(model.net.parameters(), lr=LEARNING_RATE)
    
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING) # TODO ignore pad token

    train_loss = 0
    for data_path in os.listdir(DATA_ROOT):
        dataset = PretrainDataset(data_path=DATA_ROOT/data_path, seq_len=SEQ_LEN)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        prog_bar = tqdm(enumerate(dataloader), total=(len(dataset)//BATCH_SIZE)+1)
        for batch_idx, (seq_in, seq_out) in prog_bar:
            optimizer.zero_grad()

            seq_in = seq_in.to(DEVICE)
            seq_out = seq_out.to(DEVICE)

            logits = model(seq_in)

            loss = loss_fn(logits, seq_out)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() # TODO fix so the value doesnt below

            batch_info_str = f'File {data_path}, batch {batch_idx} completed with train loss: {loss.item():.6f}'
            logger.info(batch_info_str)
            prog_bar.set_description(batch_info_str)




if __name__ == '__main__':
    pretrain()
