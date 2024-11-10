import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from rnn import RNN
import dataset

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

train = TensorDataset(train_data, train_labels)
test = TensorDataset(test_data, test_labels)

train_loader = DataLoader(train, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=test_batch_size, shuffle=True)

model = RNN(input_dim, hidden_dim, output_dim, num_layers, pred_length)
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

count = 0

for epoch in range(num_epochs):
    for batch, targets in train_loader:
        batch, targets = batch.to(device), targets.to(device)
        model.train()

        output = model(batch)

        loss = criterion(output, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        count += 1

        if count % 250 == 0:
            model.eval()
            for batch, targets in train_loader:
                batch, targets = batch.to(device), targets.to(device)
                accuracy = 0
                total = 0

                output = model(batch)

                mse = criterion(output, targets)
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
