import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import joblib
import pandas as pd
import wandb
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
    # "average",
    "volume",
    # "barCount",
    # "ema10",
    # "ema50",
]
target_features = ["high", "low"]
train_batch_size = 64
test_batch_size = 258
num_iterations = 8000
num_epochs = 80
seq_length = 15
pred_length = 8
overlap_length = 47

train_loss_list = []
iteration_list = []
test_loss_list = []

input_dim = len(input_features)
hidden_dim = 32
output_dim = len(target_features) * pred_length
num_layers = 5
lr = 1e-4

data = pd.read_csv("TQQQ15min.csv")
data = data[input_features]
data_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

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

train = TensorDataset(train_data, train_labels)
test = TensorDataset(test_data, test_labels)

train_loader = DataLoader(train, batch_size=train_batch_size, shuffle=True)
test_loader = DataLoader(test, batch_size=test_batch_size, shuffle=True)

model = RNN(input_dim, hidden_dim, output_dim, num_layers)
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.CyclicLR(
    optimizer,
    base_lr=1e-8,  # Minimum learning rate
    max_lr=1e-4,  # Maximum learning rate
    step_size_up=2500,  # Number of iterations to reach max_lr
    mode="triangular2",  # Strategy for the cyclic pattern
)

wandb.init(
    project="RNN Trader",
    config={
        "learning_rate": 0.0001,
        "architecture": "GRU",
        "dataset": "TQQQ-15min",
        "epochs": 90,
    },
)

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

        mse = criterion(outputs, targets)
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
    wandb.log({"Train Loss": loss, "Test Loss:": accuracy / total})

wandb.finish()
torch.save(model.state_dict(), "rnn_model.pth")
joblib.dump(data_scaler, "data_scaler.save")
joblib.dump(target_scaler, "target_scaler.save")
