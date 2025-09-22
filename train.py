import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from ssd_model import SSD
from utils import generate_default_boxes, decode_boxes, nms

# ====================================================
# Dataset setup
# ====================================================
IMG_DIR = "Dataset_Dog_Cat/Images"
ANN_FILE = "Dataset_Dog_Cat/annotations.json"

class CatDogDataset(Dataset):
    def __init__(self, img_dir, ann_file, transform=None):
        self.img_dir = img_dir
        self.annotations = json.load(open(ann_file))
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = list(self.annotations.keys())[idx]
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # normalize and resize
        img = cv2.resize(img, (300, 300))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # CHW format

        boxes = np.array(self.annotations[img_name]["boxes"], dtype=np.float32)
        labels = np.array(self.annotations[img_name]["labels"], dtype=np.int64)

        return torch.tensor(img), torch.tensor(boxes), torch.tensor(labels)

# ====================================================
# Training setup
# ====================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = CatDogDataset(IMG_DIR, ANN_FILE)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = SSD(num_classes=3).to(device)  # background, cat, dog
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ====================================================
# Training loop
# ====================================================
EPOCHS = 50
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for imgs, boxes, labels in dataloader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confs, locs = model(imgs)

        # dummy loss (replace with SSD MultiBox loss if needed)
        loss = criterion(confs.view(-1, 3), labels.view(-1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}] Loss: {running_loss:.4f}")

# save model
torch.save(model.state_dict(), "ssd_cat_dog.pth")
print("✅ Training complete. Model saved as ssd_cat_dog.pth")
