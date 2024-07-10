import os
import numpy as np
import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
import torchvision.models as models
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import cv2
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from sklearn.utils.class_weight import compute_class_weight


class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = csv_file
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
        image = cv2.imread(img_path)
        label = self.data.iloc[idx, 1]
        #print(f"Loaded image: {img_name} with label: {label}")
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
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ]),

    'val' : transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(256),  
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  
])}


train_csv_file = '/DATA1/tanishka/IDRID/B. Disease Grading/2. Groundtruths/a. IDRiD_Disease Grading_Training Labels.csv'
test_csv_file = '/DATA1/tanishka/IDRID/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv'
train_img_dir = '/DATA1/tanishka/IDRID/B. Disease Grading/1. Original Images/a. Training Set'
test_img_dir = '/DATA1/tanishka/IDRID/B. Disease Grading/1. Original Images/b. Testing Set'


train_data = pd.read_csv(train_csv_file)  
test_data = pd.read_csv(test_csv_file)

train_dataset = CustomDataset(train_data, train_img_dir, transform['train'])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = CustomDataset(test_data, test_img_dir, transform=transform['val'])
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

num_classes = len(train_data['Retinopathy grade'].unique())


# print("unique classes of testloader:", set(test_data['Retinopathy grade']))
# print(train_data['Retinopathy grade'].values)
# for i in range(len(test_dataset)):
#     img_name, label = test_dataset[i]
#     print(f"Image: {img_name}, Label: {label}")

model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT, progress=True)
model = model.to(device)


# freezed all parameters
for param in model.parameters():
    param.requires_grad = False

#check    
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name} is unfrozen")
    else:
        print(f"{name} is frozen")
# Unfreeze the last few convolutional layers (adjust as needed)
# for param in model.features[-3:].parameters():
#     param.requires_grad = True

# Unfreeze the classifier
# for param in model.classifier.parameters():
#     param.requires_grad = True


model.classifier = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)

optimizer = optim.Adamax(model.parameters(), lr=0.002)
# criterion = nn.CrossEntropyLoss()

y = train_data['Retinopathy grade'].values
class_weights_y = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weights_y = torch.tensor(class_weights_y, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_y)

n_epochs = 50

train_accuracies = []
test_accuracies = []

for epoch in range(1, n_epochs + 1):
    train_loss = 0.0
    correct=0.0
    model.train()
    model = model.to(device)
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)      #calculates loss using forward pass. L
        loss.backward()                     # computes gradients of loss. dL/dw, dL/db
        optimizer.step()                # updates parameters. w=w-@(dL/dw), b=b-@(dL/db)

        _,predicted=torch.max(output.data,1)
        correct += (predicted== labels).float().sum()
        accuracy = 100 * correct / len(train_loader.dataset)
        train_accuracies.append(accuracy)
        train_loss += loss.item() * images.size(0)  

    train_loss /= len(train_loader.dataset)
    print(f'Epoch {epoch}, Training Loss: {train_loss:.6f},Accuracy:{accuracy:.6f}%')



model.eval()
correct=0
total=0
with torch.no_grad():
    for images,labels in test_loader:
        images,labels=images.to(device), labels.to(device)
        output=model(images)

        _,predicted=torch.max(output.data,1)
        total+=labels.size(0)
        correct+=(predicted==labels).sum().item()
        test_accuracy = 100*correct/total
    print("Accuracy of model for test images is:",test_accuracy,"%")
    test_accuracies.append(test_accuracy)


#classwise accuracy
classes = list(set(test_data['Retinopathy grade']))
corr_pred = {classname: 0 for classname in classes}
total_pred = {classname: 0 for classname in classes}

with torch.no_grad():
  for images, labels in test_loader:
    #print("unique classes:", set(test_data['Retinopathy grade']))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    for label, pred in zip(labels, preds):
      if label == pred:
        corr_pred[classes[label.item()]] += 1
      total_pred[classes[label.item()]] += 1

for classname, corr_count in corr_pred.items():
  accuracy = 100*float(corr_count)/total_pred[classname]
  print(f'Accuracy for class: {classname:} is {accuracy:.1f} %')