import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from rnn import RNN
from stock_dataset import StockDataset

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
num_epochs = 40
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

stock_dataset = StockDataset()
train_data, train_labels, test_data, test_labels = stock_dataset.create_sequences(
    data, seq_length, pred_length, overlap_length, target_features, 0.8, True
)

train = TensorDataset(train_data, train_labels)
test = TensorDataset(test_data, test_labels)

train_loader = DataLoader(train, batch_size=train_batch_size, shuffle=False)
test_loader = DataLoader(test, batch_size=test_batch_size, shuffle=False)

model = RNN(input_dim, hidden_dim, output_dim, num_layers, pred_length, dropout)
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=1e-5,  # Minimum learning rate
    max_lr=1e-3,  # Maximum learning rate
    step_size_up=2500,  # Number of iterations to reach max_lr
    mode="triangular2",
)


count = 0

for epoch in range(num_epochs):
    for batch, targets in train_loader:
        batch, targets = batch.to(device), targets.to(device)
        model.train()
        outputs = model(batch)

        loss = torch.sqrt(criterion(outputs, targets))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        count += 1

    # eval on test data
    model.eval()
    for batch, targets in test_loader:
        batch, targets = batch.to(device), targets.to(device)
        accuracy = 0
        total = 0

        with torch.no_grad():
            outputs = model(batch)

        mse = torch.sqrt(criterion(outputs, targets))
        accuracy += mse.item()
        total += 1

    iteration_list.append(count)
    train_loss_list.append(loss)
    test_loss_list.append(accuracy / total)

    print(
        "Iteration: {}  Train Loss: {}  Test Loss: {}".format(
            count, loss, accuracy / total
        )
    )

torch.save(model.state_dict(), "rnn_model.pth")
stock_dataset.save_parameters()
