from torch import nn

class model_x(nn.Module):
    def __init__(self, in_channels, out_channels=6):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=7, kernel_size=3, stride=2, padding=1) # (48 + 2*1 - 3) / 2 + 1 = 24.5 = 24
        self.bn1 = nn.BatchNorm2d(7)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout2d(p=0.5)

        self.conv2 = nn.Conv2d(in_channels=7, out_channels=14, kernel_size=3, stride=2, padding=1)  # Corrected input channels
        self.bn2 = nn.BatchNorm2d(14)

        self.conv3 = nn.Conv2d(in_channels=14, out_channels=28, kernel_size=3, stride=2, padding=1)  # Corrected input channels
        self.bn3 = nn.BatchNorm2d(28)

        self.conv4 = nn.Conv2d(in_channels=28, out_channels=56, kernel_size=3, stride=2, padding=1) #3
        self.bn4 = nn.BatchNorm2d(56)

        self.fc1 = nn.Linear(56*3*3, 6)
        # self.fc2 = nn.Linear(128, 64)

        # self.fc3 = nn.Linear(64, 6)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.relu(out)
        # out = self.dropout(out)

        # out = self.fc2(out)
        # out = self.relu(out)

        # out = self.fc3(out)
        # out = self.relu(out)
        return out
