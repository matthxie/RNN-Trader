import torch


def create_sequences(
    data, seq_length, pred_length, overlap_length, target_features, split_percent
):
    sequences = []
    targets = []

    target_data = data[target_features].copy()

    data = torch.tensor(data.values, dtype=torch.float32)
    target_data = torch.tensor(target_data.values, dtype=torch.float32)

    for i in range(len(data) - seq_length - pred_length):
        seq = data[i : i + seq_length]
        target = target_data[i + seq_length : i + seq_length + pred_length]

        sequences.append(seq)
        targets.append(target)

    sequences = torch.stack(sequences)
    targets = torch.stack(targets)

    targets = targets.flatten(1, 2)

    split = int(sequences.shape[0] * split_percent)
    train_sequences = sequences[:split]
    train_targets = targets[:split]
    test_sequences = sequences[split:]
    test_targets = targets[split:]

    return train_sequences, train_targets, test_sequences, test_targets
