import torch
import torch.nn as nn

class CombinedNet(nn.Module):
    def __init__(self):
        super(CombinedNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.dropout_conv = nn.Dropout(p=0.25)
        self.dropout_fc = nn.Dropout(p=0.5)

        self.fc_image = nn.Linear(64 * 16 * 16, 128)

        self.fc_numeric1 = nn.Linear(163, 128)
        self.fc_numeric2 = nn.Linear(128, 128)

        self.fc_combined1 = nn.Linear(128 + 128, 64)
        self.fc_combined2 = nn.Linear(64, 32)
        self.fc_combined3 = nn.Linear(32, 1)

        self.relu = nn.ReLU()

    def forward(self, image, numeric):
        x_image = self.pool(self.relu(self.bn1(self.conv1(image))))
        x_image = self.pool(self.relu(self.bn2(self.conv2(x_image))))
        x_image = self.pool(self.relu(self.bn3(self.conv3(x_image))))
        x_image = self.dropout_conv(x_image)
        x_image = x_image.view(-1, 64 * 16 * 16)
        x_image = self.relu(self.fc_image(x_image))
        x_image = self.dropout_fc(x_image)

        x_numeric = self.relu(self.fc_numeric1(numeric))
        x_numeric = self.dropout_fc(x_numeric)
        x_numeric = self.relu(self.fc_numeric2(x_numeric))
        x_numeric = self.dropout_fc(x_numeric)

        x_combined = torch.cat((x_image, x_numeric), dim=1)
        x_combined = self.relu(self.fc_combined1(x_combined))
        x_combined = self.dropout_fc(x_combined)
        x_combined = self.fc_combined2(x_combined)
        x_combined = self.fc_combined3(x_combined)

        return x_combined

