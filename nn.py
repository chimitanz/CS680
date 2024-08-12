import numpy as np
import os
import torch
from torch import nn
from model import CombinedNet
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from PIL import Image
from matplotlib import pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mynet = CombinedNet().to(device)
loss_fn = nn.MSELoss().to(device)
optimizer = optim.SGD(mynet.parameters(), lr=0.01)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
])


class CustomDataset(Dataset):
    def __init__(self, image_folder, X_train_tensor, y_train_tensor=None, transform=None):
        self.image_folder = image_folder
        self.X_train_tensor = X_train_tensor
        self.y_train_tensor = y_train_tensor
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)

        X_train = self.X_train_tensor[idx][1:]
        if self.y_train_tensor is not None:
            y_train = self.y_train_tensor[idx]
            return image, X_train, y_train

        return image, X_train


def predict(ed, name):
    scaler = StandardScaler()
    ed = scaler.fit_transform(ed)
    mynet.load_state_dict(torch.load(name))
    ed = torch.tensor(ed, dtype=torch.float32).to(device)
    dataset = CustomDataset('./data/test_images', ed, None, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
    mynet.eval()
    all_predictions = []
    with torch.no_grad():
        for image_batch,feature_batch in dataloader:
            image_batch = image_batch.to(device)
            feature_batch = feature_batch.to(device)

            predictions = mynet(image_batch, feature_batch)

            all_predictions.append(predictions)

    all_predictions = torch.cat(all_predictions, dim=0)
    return all_predictions


def FNN(X_train, y_train, name):
    image_folder = './data/train_images'
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

    dataset = CustomDataset(image_folder, X_train_tensor, y_train_tensor, transform=transform)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    epoch = 10
    train_loss_list = []
    test_loss_list = []

    for i in range(epoch):
        print(f'---------- epoch {i + 1} for {name}----------')
        mynet.train()
        running_train_loss = 0.0
        for imgs, features, targets in train_loader:
            imgs, features, targets = imgs.to(device), features.to(device), targets.to(device)

            outputs = mynet(imgs, features)
            loss = loss_fn(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        train_loss_list.append(running_train_loss / len(train_loader))

        print(f'training loss: {train_loss_list[-1]:.4f}')
        torch.save(mynet.state_dict(), name)
        mynet.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for imgs,features, targets in test_loader:
                imgs, features, targets = imgs.to(device), features.to(device), targets.to(device)

                outputs = mynet(imgs, features)
                loss = loss_fn(outputs, targets)
                running_test_loss += loss.item()

        test_loss_list.append(running_test_loss / len(test_loader))
        print(f'test loss: {test_loss_list[-1]:.4f}')


    plt.plot(range(1, epoch + 1), train_loss_list, label='Training Loss')
    plt.plot(range(1, epoch + 1), test_loss_list, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss over Epochs for '+ name)
    plt.show()

