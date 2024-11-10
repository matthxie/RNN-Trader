import torch


def create_sequences(data, seq_length, pred_length, overlap_length):
    sequences = []
    targets = []
    for i in range(len(data) - seq_length - pred_length):
        seq = data[i : i + seq_length]
        target = data[i + seq_length : i + seq_length + pred_length]

        sequences.append(seq)
        targets.append(target)

    sequences = torch.stack(sequences)
    targets = torch.stack(targets)
    targets = targets.flatten(1, 2)

    return sequences, targets
