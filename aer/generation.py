""" Trains a time-series prediction model to generate synthetic training data. """
import argparse
import datetime

import torch
import torch.nn as nn
import numpy as np
import pandas as pd

from asta import Array, Tensor, dims, typechecked

STEPS = dims.STEPS
SEQ_LEN = dims.SEQ_LEN
IN_SIZE = dims.IN_SIZE
HIDDEN_SIZE = dims.HIDDEN_SIZE
OUT_SIZE = dims.OUT_SIZE
BATCH_SIZE = dims.BATCH_SIZE
NUM_DIRS = dims.NUM_DIRS


def parse_datetime(rep: str) -> str:
    """ Removes UTC offset. """
    segments = rep.split("+")
    if not segments:
        raise ValueError("Failed to parse datetime.")
    return segments[0]


def generate(args: argparse.Namespace) -> None:
    """ Generate synthetic training data from an example source. """
    # Assumes single-row header and format: ``<datetime>, <int>``.
    frame = pd.read_csv(args.source_path, sep=",")
    raw_series = frame.values
    dims.STEPS = len(raw_series)

    # Convert strings to datetimes.
    for i in range(dims.STEPS):
        stamp = parse_datetime(raw_series[i][0])
        raw_series[i][0] = datetime.datetime.strptime(stamp, "%Y-%m-%d %H:%M:%S")

    series: Array[STEPS] = raw_series[:, 1]

    # Normalize the entire time series.
    mean, std = np.mean(series), np.std(series)
    series = series - mean
    series = series / std

    # Arrange in sequences for training (-1 is so we have targets for each seq).
    num_seqs = max(0, dims.STEPS - args.seq_len - 1)

    # Shapes.
    sequences: Array[int, num_seqs, args.seq_len]
    targets: Array[int, num_seqs]

    # Sequences and their targets.
    sequences = np.zeros((num_seqs, args.seq_len))
    targets = np.zeros((num_seqs,))

    # Set values for sequence and target arrays.
    for i in range(num_seqs):
        sequences[i] = series[i : i + args.seq_len]
        targets[i] = series[i + args.seq_len + 1]

    # Batch the sequences and targets.
    num_batches = num_seqs // args.batch_size
    sequences = sequences[: num_batches * args.batch_size]
    targets = targets[: num_batches * args.batch_size]

    sequences = sequences.reshape(num_batches, args.batch_size, args.seq_len, 1)
    targets = targets.reshape(num_batches, args.batch_size, 1)

    # Create the model.
    model = LSTM(1, args.hidden_size, 1, args.batch_size, args.num_layers, 0.5, True)
    optimizer = torch.optim.Adam(model.parameters())
    loss_fn = nn.MSELoss()

    train_sequences = torch.FloatTensor(sequences)
    train_targets = torch.FloatTensor(targets)

    batch: Tensor[float, (args.batch_size, args.seq_len, 1)]
    target: Tensor[float, (args.batch_size, 1)]
    pred: Tensor[float, (args.batch_size, 1)]

    # The main training loop.
    for _ in range(args.epochs):
        losses = []
        for batch, target in zip(train_sequences, train_targets):

            optimizer.zero_grad()
            pred = model(batch)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        epoch_loss = np.mean(losses)
        print("Loss:", epoch_loss)

    # The generation/eval loop.
    model.eval()
    batch = train_sequences[-1]
    for _ in range(10000):
        with torch.no_grad():
            # Make prediction.
            pred = model(batch)

            # Flatten the batch, convert to a list, remove first element.
            flat_batch = list(batch.view(-1).numpy())
            flat_batch = flat_batch[-len(flat_batch) + 1:]

            # Append prediction to flat batch.
            flat_batch.append(pred[-1][0])

            # Convert back to appropriately-shaped tensor.
            batch = torch.Tensor(flat_batch).view(args.batch_size, args.seq_len, 1)

            print(pred[-1][0])

class LSTM(nn.Module):
    """ A simple LSTM module. """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        batch_size: int,
        num_layers: int,
        dropout: float,
        bi: bool,
    ):
        super().__init__()
        num_dirs = 2 if bi else 1
        self.batch_size = batch_size
        self.output_size = output_size
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bi,
        )
        self.linear = nn.Linear(num_dirs * hidden_size, output_size)

        # LSTM states.
        self.hidden = torch.zeros((num_layers * num_dirs, batch_size, hidden_size))
        self.cell = torch.zeros((num_layers * num_dirs, batch_size, hidden_size))

    def forward(
        self, x: Tensor[float, (BATCH_SIZE, SEQ_LEN, IN_SIZE)]
    ) -> Tensor[float, (BATCH_SIZE, OUT_SIZE)]:
        """ Execute a forward pass through the module. """
        self.hidden.detach_()
        self.cell.detach_()

        logits: Tensor[float, (BATCH_SIZE, SEQ_LEN, NUM_DIRS * HIDDEN_SIZE)]

        # Pass the input through the LSTM layers.
        logits, states = self.lstm(x, (self.hidden, self.cell))

        # Update the LSTM states.
        self.hidden, self.cell = states

        # Pass the LSTM last layer logits/outs through linear layer.
        outs: Tensor[float, (BATCH_SIZE, SEQ_LEN, OUT_SIZE)] = self.linear(logits)

        # Return the last output of every every sequence in the batch.
        preds = outs[:, -1, :].view(self.batch_size, self.output_size)

        return preds
