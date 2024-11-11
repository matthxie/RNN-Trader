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
hidden_dim = 32
output_dim = len(target_features) * pred_length
num_layers = 2
lr = 1e-4

data = pd.read_csv("TQQQ15min.csv")
data = data[input_features]

train_data, train_labels, test_data, test_labels = dataset.create_sequences(
    data, seq_length, pred_length, overlap_length, target_features, 0.8
)

train = TensorDataset(train_data, train_labels)
test = TensorDataset(test_data, test_labels)

train_loader = DataLoader(train, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=test_batch_size, shuffle=True)

model = RNN(input_dim, hidden_dim, output_dim, num_layers)
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

count = 0

for epoch in range(num_epochs):
    for batch, targets in train_loader:
        batch, targets = batch.to(device), targets.to(device)
        model.train()

        outputs = model(batch)

        loss = criterion(outputs, targets)

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

                with torch.no_grad():
                    outputs = model(batch)

                mse = criterion(outputs, targets)
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

torch.save(model.state_dict(), "rnn_model.pth")
