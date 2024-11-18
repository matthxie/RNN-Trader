import torch
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class StockDataset:

    def __init__(self):
        self.input_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

        self.lambda_inputs = 0
        self.lambda_targets = 0

        self.train_test_split_index = 0

    def create_sequences(
        self,
        data,
        seq_length,
        pred_length,
        overlap_length,
        target_features,
        split_percent,
        shuffle=True,
    ):
        sequences = []
        targets = []

        self.train_test_split_index = int(data.shape[0] * split_percent)

        target_data = data[target_features].copy()

        data = torch.tensor(data.values, dtype=torch.float32)
        target_data = torch.tensor(target_data.values, dtype=torch.float32)

        data, target_data = self.transform(data, target_data)

        for i in range(len(data) - seq_length - pred_length):
            seq = data[i : i + seq_length]
            target = target_data[i + seq_length : i + seq_length + pred_length]

            sequences.append(seq)
            targets.append(target)

        sequences = torch.stack(sequences)
        targets = torch.stack(targets)

        if shuffle:
            indices = torch.randperm(sequences.size(0))
            sequences = sequences[indices]
            targets = targets[indices]

        split = self.train_test_split_index
        train_sequences = sequences[:split]
        train_targets = targets[:split]
        test_sequences = sequences[split:]
        test_targets = targets[split:]

        return train_sequences, train_targets, test_sequences, test_targets

    def transform(self, inputs, targets):
        inputs = np.where(inputs > 0.0000000001, inputs, 1e-10)
        targets = np.where(targets > 0.0000000001, targets, 1e-10)

        inputs = np.log(inputs)
        targets = np.log(targets)

        inputs = torch.from_numpy(self.input_scaler.fit_transform(inputs)).to(
            dtype=torch.float32
        )
        targets = torch.from_numpy(self.target_scaler.fit_transform(targets)).to(
            dtype=torch.float32
        )

        return inputs, targets

    def inverse_transform_inputs(self, inputs):
        inputs = self.input_scaler.inverse_transform(inputs)
        inputs = np.exp(inputs)

        return inputs

    def inverse_transform_targets(self, targets):
        targets = self.target_scaler.inverse_transform(targets)
        targets = np.exp(targets)

        return targets

    def save_parameters(self):
        joblib.dump(self, "transform.save")

    def load_parameters(self):
        return joblib.load("transform.save")
