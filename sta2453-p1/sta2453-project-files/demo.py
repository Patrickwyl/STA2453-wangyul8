import pandas as pd
import torch
from torch import nn
from torch import optim


class PutNet(nn.Module):
    """
    Example of a Neural Network that could be trained price a put option.
    TODO: modify me!
    """

    def __init__(self) -> None:
        super(PutNet, self).__init__()

        self.layers = nn.Sequential(nn.BatchNorm1d(5),
                                    nn.Linear(5, 500),
                                    #nn.BatchNorm1d(400),
                                    nn.ReLU(),
                                    nn.Linear(500, 500),
                                    #nn.BatchNorm1d(400),
                                    nn.ReLU(),
                                    nn.Linear(500, 500),
                                    #nn.BatchNorm1d(400),
                                    nn.ReLU(),
                                    nn.Linear(500, 500),
                                    # nn.BatchNorm1d(400),
                                    nn.ReLU(),
                                    nn.Linear(500, 1))

    def forward(self, x):
        return self.layers(x)


def main():
    """Train the model and save the checkpoint"""

    # Create model
    model = PutNet()

    # Load training and validation dataset
    df = pd.read_csv("training.csv")
    val_df = pd.read_csv("validation.csv")

    # Set up training
    x = torch.Tensor(df[["S", "K", "T", "r", "sigma"]].to_numpy())
    y = torch.Tensor(df[["value"]].to_numpy())

    # Set up validation
    x_val = torch.Tensor(val_df[["S", "K", "T", "r", "sigma"]].to_numpy())
    y_val = torch.Tensor(val_df[["value"]].to_numpy())

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)  # Use Adam optimizer

    # Train for 1000 epochs
    for i in range(1000):

        y_hat = model(x)
        y = y

        # Calculate training loss
        training_loss = criterion(y_hat, y)

        # Take a step
        optimizer.zero_grad()
        training_loss.backward()
        optimizer.step()

        # Check validation loss
        with torch.no_grad():
            validation_loss = criterion(model(x_val), y_val)

        print(f"Iteration: {i} | Training Loss: {training_loss:.4f} | Validation Loss: {validation_loss:.4f} ")

    torch.save(model.state_dict(), "simple-model.pt")


if __name__ == "__main__":
    main()
