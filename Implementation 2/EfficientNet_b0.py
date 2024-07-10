import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
from torch.optim import Adam
import torchvision.models as models
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.models import EfficientNet_B0_Weights
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
        image = cv2.imread(img_path)
        label = self.data.iloc[idx, 1]
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if self.transform:
            image = self.transform(image)

        return image, label

transform = {
    'train' : transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(256),  
        transforms.RandomRotation(degrees=45), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomHorizontalFlip(),  
        transforms.RandomVerticalFlip(), 
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ]),

    'val' : transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(256),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

train_csv_file = '/DATA1/tanishka/IDRID/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv'
test_csv_file = '/DATA1/tanishka/IDRID/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv'
train_img_dir = '/DATA1/tanishka/IDRID/B. Disease Grading/1. Original Images/a. Training Set'
test_img_dir = '/DATA1/tanishka/IDRID/B. Disease Grading/1. Original Images/b. Testing Set'

train_dataset = CustomDataset(train_csv_file, train_img_dir, transform['train'])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = CustomDataset(test_csv_file, test_img_dir, transform=transform['val'])
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

train_data = pd.read_csv(train_csv_file)  
num_classes = len(train_data['Retinopathy grade'].unique())


model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT, progress=True)
model = model.to(device)

for param in model.parameters():
    param.requires_grad = False

# for param in model.features[-3:].parameters():
#     param.requires_grad = True

model.classifier = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.4),
    nn.Linear(512, num_classes)
)

optimizer = optim.Adamax(model.parameters(),lr=0.002)

y = train_data['Retinopathy grade'].values
class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
n_epochs = 100

for epoch in range(1, n_epochs + 1):
    train_loss = 0.0
    correct = 0.0
    model.train()
    model = model.to(device)
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output.data, 1)
        correct += (predicted == labels).float().sum()
        accuracy = 100 * correct / len(train_loader.dataset)
        
        train_loss += loss.item() * images.size(0)

    train_loss /= len(train_loader.dataset)
    print(f'Epoch {epoch}, Training Loss: {train_loss:.6f}, Accuracy: {accuracy:.6f}')

model.eval()
correct = 0
total = 0

all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        output = model(images)

        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())

    print("Accuracy of model for test images is: {}%".format(100 * correct / total))

# Classification report
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=[str(i) for i in range(num_classes)]))

# Confusion matrix
def plot_confusion_matrix(labels, pred_labels):

    target_names = [ '0', '1','2','3', '4']

    print(classification_report(labels, pred_labels,target_names=target_names))

    #print(accuracy_score(labels, pred_labels))

    ConfusionMatrix = confusion_matrix(labels, pred_labels)

    print("confusion matrix on test set")

    print(ConfusionMatrix) 

    CM = ConfusionMatrix.astype('float') / ConfusionMatrix.sum(axis=1)[:, np.newaxis]

    print("classwise accuracy on test set of model",CM.diagonal()*100)

    return None

plot_confusion_matrix(all_labels, all_preds)


torch.save(model.state_dict(), "efficientnet_b0_retinopathy.pth")
print("Model saved as efficientnet_b0_retinopathy.pth")

