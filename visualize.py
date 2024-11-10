import torch
import pandas as pd
import matplotlib.pyplot as plt
from random import randint
from rnn import RNN

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    # else (torch.device("mps") if torch.mps.is_available() else torch.device("cpu"))
    else torch.device("cpu")
)

num_features = 5
train_batch_size = 100
test_batch_size = 100
num_iterations = 8000
num_epochs = 4
seq_length = 48
pred_length = 10
overlap_length = 47

seq_dim = 28
loss_list = []
iteration_list = []
accuracy_list = []

input_dim = 5
hidden_dim = 20
output_dim = 5
num_layers = 1

data = pd.read_csv("TQQQ15min.csv")
data = data[["open", "high", "low", "close", "volume"]]
data = torch.tensor(data.values, dtype=torch.float32)

model = RNN(input_dim, hidden_dim, output_dim, num_layers, pred_length)
model.load_state_dict(torch.load("rnn_model.pth", weights_only=True))
model.eval()
model.to(device)

index = randint(0, data.shape[0] - seq_length - pred_length)

input = data[index : index + seq_length]
label = data[index + seq_length : index + seq_length + pred_length]
with torch.no_grad():
    prediction = model(input.unsqueeze(0).to(device)).squeeze(0)
prediction = prediction.view(-1, 5)


print(input.shape, ", ", label.shape, ", ", prediction.shape)

plt.plot(range(48), input, label="Input Sequence")
plt.plot(range(48, 58), label, label="True Future", color="blue")
plt.plot(
    range(48, 58),
    prediction,
    label="Predicted Future",
    color="orange",
    linestyle="dashed",
)
plt.xlabel("Time Steps")
plt.ylabel("Value")
plt.legend()
plt.title("Sequential Data Prediction using RNN")
plt.show()
