import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_data, num_classes) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_data, 200),
            nn.ReLU(True),
            nn.Linear(200, 200),
            nn.ReLU(True),
            nn.Linear(200, num_classes),
            nn.ReLU(True),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        x = x.squeeze(dim=1)
        x = self.model(x)
        return x