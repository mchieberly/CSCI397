from torch import nn
import copy

"""
Class for Mario online and target neural nets
"""
class MarioNet(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        # Initialize height and width with input_dim
        c, h, w = input_dim

        if h != 84:
            raise ValueError(f"Expecting input height: 84, got: {h}")
        if w != 84:
            raise ValueError(f"Expecting input width: 84, got: {w}")

        # Define the online CNN, using the proper inputs and the ReLU function
        self.online = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )

        # Create the target network as a copy of the online network
        self.target = copy.deepcopy(self.online)

        for p in self.target.parameters():
            p.requires_grad = False

    # Separate forward for online and target networks
    def forward(self, input, model):
        if model == 'online':
            return self.online(input)
        elif model == 'target':
            return self.target(input)
