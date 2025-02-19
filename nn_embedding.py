import torch
import torch.nn as nn


class MINE(nn.Module):
    def __init__(self):
        super(MINE, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, y):
        input_xy = torch.stack([x.float(), y.float()], dim=1)
        return self.net(input_xy)


def compute_llr(model_path, y_values, device="cpu"):
    """
    Computes the Log-Likelihood Ratio (LLR) using the trained MINE model.

    Parameters:
        model_path (str): Path to the trained model file.
        y_values (torch.Tensor or list): Observed values of Y.
        device (str): Device to use ("cpu" or "cuda").

    Returns:
        torch.Tensor: LLR values for each y in y_values.
    """
    # Load the model
    model = MINE()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval().to(device)

    # Convert y_values to tensor if needed
    if isinstance(y_values, list):
        y_values = torch.tensor(y_values, dtype=torch.float32, device=device)
    elif not isinstance(y_values, torch.Tensor):
        raise TypeError("y_values must be a list or a torch.Tensor")

    # Generate X values: X=1 and X=0
    ones = torch.ones_like(y_values, device=device)
    zeros = torch.zeros_like(y_values, device=device)

    # Compute T(1, y) and T(0, y)
    T_1y = model(ones, y_values).squeeze()
    T_0y = model(zeros, y_values).squeeze()

    # Compute LLR as T(1, y) - T(0, y)
    LLR = T_0y - T_1y

    return LLR


# Example usage
if __name__ == "__main__":
    model_path = "mine_trained_on_N0_range.pth"
    y_values = [-1.0, 0.0, 1.0]  # Example Y values
    llr_values = compute_llr(model_path, y_values)
    print("LLR values:", llr_values.tolist())
