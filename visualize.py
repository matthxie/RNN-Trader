import torch
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import matplotlib.pyplot as plt
from random import randint
import joblib
from rnn import RNN
import dataset

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    # else (torch.device("mps") if torch.mps.is_available() else torch.device("cpu"))
    else torch.device("cpu")
)

input_features = [
    "open",
    "high",
    "low",
    "close",
    "average",
    "volume",
    # "barCount",
    "ema10",
    "ema50",
]
target_features = ["high", "low"]
train_batch_size = 64
test_batch_size = 1024
num_epochs = 30
seq_length = 15
pred_length = 3
overlap_length = 47

train_loss_list = []
iteration_list = []
test_loss_list = []

input_dim = len(input_features)
hidden_dim = 64
output_dim = len(target_features)
num_layers = 2
dropout = 0.0
lr = 1e-3

data = pd.read_csv("TQQQ15min.csv")
data = data[input_features]

data_scaler = joblib.load("data_scaler.save")
target_scaler = joblib.load("target_scaler.save")

train_data, train_labels, test_data, test_labels = dataset.create_sequences(
    data,
    seq_length,
    pred_length,
    overlap_length,
    target_features,
    0.8,
    data_scaler,
    target_scaler,
)

model = RNN(input_dim, hidden_dim, output_dim, num_layers, pred_length, dropout)
model.load_state_dict(torch.load("rnn_model.pth", map_location=device))
model.eval()
model.to(device)

criterion = torch.nn.MSELoss(reduction="mean")

test = TensorDataset(test_data, test_labels)
test_loader = DataLoader(test, batch_size=1, shuffle=True)

count = 0
for input, label in test_loader:
    input, label = input.to(device), label.to(device)
    model.eval()

    with torch.no_grad():
        prediction = model(input)

    loss = criterion(prediction, label)

    print(loss)

    prediction = torch.squeeze(prediction).to("cpu")
    label = torch.squeeze(label).to("cpu")
    input = torch.squeeze(input).to("cpu")

    print(prediction)
    print(label)

    prediction = target_scaler.inverse_transform(prediction)
    input = data_scaler.inverse_transform(input)
    label = target_scaler.inverse_transform(label)

    print(prediction)
    print(label)

    plt.figure(figsize=(10, 6))

    plt.plot(range(seq_length), input[:, 0], label="Input Sequence - low", color="gray")
    plt.plot(
        range(seq_length, seq_length + pred_length),
        label[:, 0],
        label="True Future - high",
        color="green",
    )
    plt.plot(
        range(seq_length, seq_length + pred_length),
        prediction[:, 0],
        label="Predicted Future - high",
        color="green",
        linestyle="dashed",
    )

    plt.plot(
        range(seq_length), input[:, 1], label="Input Sequence - high", color="lightgray"
    )
    plt.plot(
        range(seq_length, seq_length + pred_length),
        label[:, 1],
        label="True Future - low",
        color="red",
    )
    plt.plot(
        range(seq_length, seq_length + pred_length),
        prediction[:, 1],
        label="Predicted Future - low",
        color="red",
        linestyle="dashed",
    )

    plt.xlabel("Time Steps")
    plt.ylabel("Value")
    plt.title("Sequential Data Prediction using RNN")
    plt.legend()
    plt.show()

    count += 1
    if count > 10:
        break
