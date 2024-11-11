import torch
import pandas as pd
import matplotlib.pyplot as plt
from random import randint
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
    "volume",
    "average",
    "barCount",
    "ema10",
    "ema50",
]
target_features = ["high", "low"]
train_batch_size = 100
test_batch_size = 100
num_iterations = 8000
num_epochs = 8
seq_length = 48
pred_length = 10
overlap_length = 47

loss_list = []
iteration_list = []
accuracy_list = []

input_dim = len(input_features)
hidden_dim = 20
output_dim = len(target_features) * pred_length
num_layers = 2
lr = 1e-4

data = pd.read_csv("TQQQ15min.csv")
data = data[input_features]

train_data, train_labels, test_data, test_labels = dataset.create_sequences(
    data, seq_length, pred_length, overlap_length, target_features, 0.8
)

model = RNN(input_dim, hidden_dim, output_dim, num_layers)
model.load_state_dict(torch.load("rnn_model.pth", weights_only=True))
model.eval()
model.to(device)

index = randint(0, test_data.shape[0] - seq_length - pred_length)

input = test_data[index]
label = test_labels[index]


with torch.no_grad():
    prediction = model(input.unsqueeze(0).to(device)).to("cpu")

label = label.view(-1, len(target_features))
prediction = prediction.view(-1, len(target_features))

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
