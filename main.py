import torch
import torch.nn as nn
from rnn import RNN
import dataset
import pandas as pd

device = (
    torch.device("cuda")
    if torch.cuda.is_available()
    # else (torch.device("mps") if torch.mps.is_available() else torch.device("cpu"))
    else torch.device("cpu")
)

num_features = 5
batch_size = 100
num_iterations = 8000
num_epochs = 3
seq_length = 48
pred_length = 10
overlap_length = 47

seq_dim = 28
loss_list = []
iteration_list = []
accuracy_list = []
count = 0

input_dim = 5
hidden_dim = 10
output_dim = 5
num_layers = 1
lr = 1e-3

data = pd.read_csv("TQQQ15min.csv")
data = data[["open", "high", "low", "close", "volume"]]
data = torch.tensor(data.values, dtype=torch.float32)

split = int(data.shape[0] * 0.8)
train_data, train_labels = dataset.create_sequences(
    data[:split], seq_length, pred_length, overlap_length
)
test_data, test_labels = dataset.create_sequences(
    data[split:], seq_length, pred_length, overlap_length
)

model = RNN(input_dim, hidden_dim, output_dim, num_layers, pred_length)

train_data.to(device)
train_labels.to(device)
test_data.to(device)
test_labels.to(device)
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

count = 0

for epoch in range(num_epochs):
    for i in range(train_data.shape[0]):
        model.train()
        optimizer.zero_grad()

        output = model(train_data[i])

        loss = criterion(output, train_labels[i])
        loss.backward()
        optimizer.step()

        count += 1

        if count % 250 == 0:
            model.eval()
            for j in range(test_data.shape[0]):
                accuracy = 0
                total = 0

                output = model(test_data[j])

                mse = criterion(output, test_labels[j])
                accuracy += mse.item()
                total += 1

            iteration_list.append(count)
            loss_list.append(loss)
            accuracy_list.append(accuracy / total)

            print(
                "Iteration: {}  Train Loss: {}  Test MSE: {}".format(
                    count, loss, accuracy / total
                )
            )
