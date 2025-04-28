import torch
from torch import nn
from helpers.matrix import create_data_matrix
from helpers.edge_tensor import edges
from model import TemporalGNNModel
from torch_geometric_temporal.signal import temporal_signal_split, StaticGraphTemporalSignal
from torch.optim import Adam
from hyperparameters import LR, BETAS, EPOCHS, HORIZON, DEVICE
import wandb


def train_snapshot():
    temporal_loss = 0
    for step, snapshot in enumerate(train):
        snapshot = snapshot.to(device=device)
        output = model(snapshot.x, snapshot.edge_index)

        loss = loss_function(output, snapshot.y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        temporal_loss += loss.item()

    return temporal_loss/(step+1)


@torch.no_grad()
def val_snapshot():
    temporal_loss = 0
    for step, snapshot in enumerate(val):
        snapshot = snapshot.to(device=device)
        output = model(snapshot.x, snapshot.edge_index)

        loss = loss_function(output, snapshot.y)
        temporal_loss += loss.item()

    return temporal_loss/(step+1)


def training():
    for epoch in range(EPOCHS):
        model.train(True)
        train_loss = train_snapshot()

        model.eval()
        val_loss = val_snapshot()

        print(f"Epoch: {epoch+1}")
        print(f"Train Loss: {train_loss}")
        print(f"Validation Loss: {val_loss}")

        wandb.log({
            "Train Loss": train_loss,
            "Validation Loss": val_loss
        })

        if (epoch+1) % 10 == 0:
            save_path = f"/Volumes/Vault/Smudge/IIT_Data/Motion/Checkpoints/TGCN/{epoch+1}.pth"
            torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    device = DEVICE

    data_matrix = create_data_matrix()

    features_matrix = data_matrix[:-HORIZON]  # Horizon->Prediction steps
    targets_matrix = data_matrix[HORIZON:]

    edge_weight_tensor = torch.ones(size=(edges.size(0),))
    temporal_loader = StaticGraphTemporalSignal(
        edge_index=edges, features=features_matrix, targets=targets_matrix, edge_weight=edge_weight_tensor)

    train, val = temporal_signal_split(temporal_loader, 0.8)

    model = TemporalGNNModel().to(device=device)
    loss_function = nn.MSELoss()
    optimizer = Adam(params=model.parameters(), lr=LR, betas=BETAS)

    wandb.init(
        project="Graph Motion using Temporal GNNs"
    )

    training()
