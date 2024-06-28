import os
import cv2
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool
import matplotlib.pyplot as plt


# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DUTSDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, target_size=(256, 256)):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.target_size = target_size
        self.valid_samples = self._find_valid_samples()

    def __len__(self):
        return len(self.valid_samples)

    def __getitem__(self, idx):
        img_path = self.valid_samples[idx]['image_path']
        mask_path = self.valid_samples[idx]['mask_path']

        image = cv2.imread(img_path)
        if image is None:
            return self.__getitem__((idx + 1) % len(self))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.target_size)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return self.__getitem__((idx + 1) % len(self))

        mask = cv2.resize(mask, self.target_size)

        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask).unsqueeze(0)

        return image, mask

    def _find_valid_samples(self):
        valid_samples = []
        for idx, row in self.data.iterrows():
            img_path = os.path.join(self.root_dir, row['image_path'])
            mask_path = os.path.join(self.root_dir, row['mask_path'])
            if os.path.exists(img_path) and os.path.exists(mask_path):
                row_with_full_paths = row.copy()
                row_with_full_paths['image_path'] = img_path
                row_with_full_paths['mask_path'] = mask_path
                valid_samples.append(row_with_full_paths)
        return valid_samples

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor()
])

metadata_file = '/home/mayun/download/nuist_projects/DUTS_light/metadata.csv'
root_directory = '/home/mayun/download/nuist_projects/DUTS_light/'

duts_dataset = DUTSDataset(csv_file=metadata_file, root_dir=root_directory, transform=transform)
duts_dataloader = DataLoader(duts_dataset, batch_size=4, shuffle=True)

class GNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 32 * 32, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class GNN_CNN_Saliency(nn.Module):
    def __init__(self, gnn_in_channels, gnn_hidden_channels, gnn_out_channels, cnn_out_features, num_classes):
        super(GNN_CNN_Saliency, self).__init__()
        self.gnn = GNN(gnn_in_channels, gnn_hidden_channels, gnn_out_channels)
        self.cnn = CNN()
        self.fc = nn.Linear(gnn_out_channels + cnn_out_features, num_classes)

    def forward(self, data, image):
        gnn_out = self.gnn(data.x, data.edge_index, data.batch)
        cnn_out = self.cnn(image)
        combined = torch.cat([gnn_out, cnn_out], dim=1)
        return self.fc(combined).view(combined.size(0), 1, 256, 256)

num_node_features = 16
edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long).to(device)

def create_gnn_data_batch(batch_size, num_nodes, num_node_features):
    data_list = []
    for _ in range(batch_size):
        node_features = torch.randn((num_nodes, num_node_features)).to(device)
        batch = torch.zeros(num_nodes, dtype=torch.long).to(device)
        data = Data(x=node_features, edge_index=edge_index, batch=batch)
        data_list.append(data)
    return Batch.from_data_list(data_list)

model = GNN_CNN_Saliency(gnn_in_channels=num_node_features, gnn_hidden_channels=32, gnn_out_channels=64, cnn_out_features=512, num_classes=256 * 256).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

def show_image(img, title="Image"):
    img = img.numpy().transpose((1, 2, 0))
    plt.imshow(img)
    plt.title(title)
    plt.show()

loss_values = []

for epoch in range(40):
    model.train()
    for batch_idx, (images, masks) in enumerate(duts_dataloader):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        batch_size = images.size(0)
        num_nodes = 4
        gnn_data_batch = create_gnn_data_batch(batch_size, num_nodes, num_node_features)

        output = model(gnn_data_batch, images)
        masks = masks.float()
        loss = criterion(output, masks.squeeze(1))
        loss.backward()
        optimizer.step()

        loss_values.append(loss.item())

        print(f'Epoch {epoch + 1}, Batch {batch_idx + 1}, Loss: {loss.item()}')

    # Save the model weights after each epoch
    model_path = f'./weights/model_weights_epoch_{epoch + 1}.pth'
    torch.save(model.state_dict(), model_path)
    print(f'Model weights saved to {model_path} after epoch {epoch + 1}')

    if (epoch + 1) % 20 == 0:
        print(f'Processed {batch_idx + 1} batches in epoch {epoch + 1}')
        show_image(images[0].cpu(), title="Input Image")
        show_image(masks[0].squeeze(0).cpu(), title="Ground Truth Mask")
        show_image(output[0].sigmoid().detach().cpu(), title="Predicted Mask")

# Plot the loss values
plt.figure()
plt.plot(loss_values, label='Training Loss')
plt.xlabel('Batch Number')
plt.ylabel('Loss')
plt.title('Training Loss Over Time')
plt.legend()
plt.show()

model.eval()
with torch.no_grad():
    for batch_idx, (images, masks) in enumerate(duts_dataloader):
        images, masks = images.to(device), masks.to(device)
        batch_size = images.size(0)
        gnn_data_batch = create_gnn_data_batch(batch_size, num_nodes, num_node_features)
        output = model(gnn_data_batch, images)
        print(output.cpu())
