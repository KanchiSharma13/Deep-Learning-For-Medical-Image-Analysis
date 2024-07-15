import os
import cv2
import numpy as np
import torch
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.models as models
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#create custom dataloader
class CustomDataset(Dataset):
    def __init__(self, img_dir, csv_file, transform):
        self.img_dir = img_dir
        self.csv_file = csv_file
        self.transform = transform

    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, idx):
        img_name = self.csv_file.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, f"{img_name}.jpg")
        image = cv2.imread(img_path)
        label = self.csv_file.iloc[idx, 1]
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if self.transform:
            image = self.transform(image)
        return image, label
      
#add transformations to images
transform = {
    'train': transforms.Compose([
        transforms.CenterCrop((900, 900)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.CenterCrop((900, 900)),  
        transforms.ToTensor(),  
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

train_csv_file = '/workspace/Desktop/IDRID1/IDRID/B. Disease Grading/2. Groundtruths/train.csv'
train_img_dir = '/workspace/Desktop/IDRID1/IDRID/B. Disease Grading/1. Original Images/train'
# test_csv_file = '/DATA1/tanishka/IDRID/B. Disease Grading/2. Groundtruths/b. IDRiD_Disease Grading_Testing Labels.csv'
# test_img_dir = '/DATA1/tanishka/IDRID/B. Disease Grading/1. Original Images/b. Testing Set'

train_data = pd.read_csv(train_csv_file)
# test_data=pd.read_csv(test_csv_file)

train_dataset = CustomDataset(train_img_dir,train_data, transform['train'])
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataset = CustomDataset(train_img_dir,train_data ,transform=transform['val'])
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

#for ResNet5o:
num_classes = len(set(train_data['Retinopathy grade']))
model = models.resnet50(weights='DEFAULT', progress=True)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

#for VGG:
# model = models.vgg19(weights='DEFAULT', progress=True)
# model.fc = torch.nn.Linear(model.classifier[6].in_features, num_classes)

# for alexnet:
# model = models.alexnet(weights='DEFAULT', progress=True)
# model.fc = torch.nn.Linear(model.classifier[6].in_features, num_classes)

#transfer learning
#freeze all layers
for param in model.parameters():
    param.requires_grad = False
#unfreeze last fc layer
for param in model.fc.parameters():
    param.requires_grad = True

#Cross-Validation
skf = StratifiedKFold(n_splits=5, shuffle=True)
fold_accuracies = []
all_labels = []
all_preds = []
all_preds_prob = []

for fold, (train_index, val_index) in enumerate(skf.split(train_data, train_data['Retinopathy grade'])):
    print(f"Split: {fold + 1}")

    train_subset = Subset(train_data, train_index)
    val_subset = Subset(train_data, val_index)

    train_dataset = CustomDataset(train_img_dir, train_data.iloc[train_index], transform['train'])
    val_dataset = CustomDataset(train_img_dir, train_data.iloc[val_index], transform['val'])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
   
    
    # Stage 1: Training with Adadelta

    optimizer_adadelta = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    best_loss = float('inf')
    best_model_weights = None
    for epoch in range(20):  
        model.train()
        running_loss = 0.0
        correct = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer_adadelta.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_adadelta.step()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).float().sum()
            accuracy = 100 * correct / len(train_loader.dataset)

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_weights = model.state_dict()

    model.load_state_dict(best_model_weights)

    
    #unfreeze layer 4
    for param in model.layer4.parameters():
        param.requires_grad = True
    # Stage 2: Fine-tuning with SGD
  
    #check
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name} is unfrozen")
    #     else:
    #         print(f"{name} is frozen")
    optimizer = optim.SGD(model.parameters(), lr=0.001)
    best_loss = float('inf')
    best_model_weights = None
    
    
    for epoch in range(50):
        model.train()
        running_loss = 0.0
        correct = 0
        patience = 20
        no_improve = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).float().sum()
            accuracy = 100 * correct / len(train_loader.dataset)

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f'Epoch {epoch + 1}, Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_model_weights = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            print("Early stopping")
            break

    model.load_state_dict(best_model_weights)
    torch.save(model.state_dict(), "ResNet-50b FINAL.pth")
  
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    fold_labels = []
    fold_preds = []
    fold_preds_prob = []

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            fold_labels.extend(labels.cpu().numpy())
            fold_preds.extend(predicted.cpu().numpy())
            fold_preds_prob.extend(probabilities.cpu().numpy())

    all_labels.extend(fold_labels)
    all_preds.extend(fold_preds)
    all_preds_prob.extend(fold_preds_prob)

    fold_accuracy = 100 * correct / total
    fold_accuracies.append(fold_accuracy)
    #print(f"Fold {fold + 1} Accuracy: {fold_accuracy:.2f}%")
average_accuracy = np.mean(fold_accuracies)
print(f"Average Accuracy across all folds: {average_accuracy:.2f}%")   

# Convert to numpy arrays
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_preds_prob = np.array(all_preds_prob)

# print(all_preds_prob)

#plot roc curve
def plot_multiclass_roc_with_hist(y_true, y_pred_proba, n_classes, class_names):
    
    y_true_bin = label_binarize(y_true, classes=range(n_classes))

    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    
    fig, axs = plt.subplots(2, n_classes, figsize=(5*n_classes, 10))
    
    for i in range(n_classes):
        # Plot ROC curve
        axs[0, i].plot(fpr[i], tpr[i], lw=2)
        axs[0, i].plot([0, 1], [0, 1], 'k--', lw=2)
        axs[0, i].set_xlim([0.0, 1.0])
        axs[0, i].set_ylim([0.0, 1.05])
        axs[0, i].set_xlabel('False Positive Rate')
        axs[0, i].set_ylabel('True Positive Rate')
        axs[0, i].set_title(f'ROC Curve for {class_names[i]}\nAUC = {roc_auc[i]:.2f}')

        # Plot histogram
        axs[1, i].hist(y_pred_proba[y_true == i, i], bins=20, alpha=0.5, label='Positive')
        axs[1, i].hist(y_pred_proba[y_true != i, i], bins=20, alpha=0.5, label='Negative')
        axs[1, i].set_xlabel('Predicted Probability')
        axs[1, i].set_ylabel('Frequency')
        axs[1, i].set_title(f'Histogram for {class_names[i]}')
        axs[1, i].legend()

    plt.tight_layout()
    plt.savefig('multiclass_roc_and_hist_resnet50b FINAL.png', dpi=300, bbox_inches='tight')
    plt.close()
    return roc_auc

n_classes = 5
class_names = ['Class 0', 'Class 1','Class 2', 'Class 3', 'Class 4']

plot_multiclass_roc_with_hist(all_labels, all_preds_prob, num_classes, class_names)
print("Multiclass ROC curves and histograms saved as 'multiclass_roc_and_hist Resnet-50b FINAL.png'")


#plot confusion matrix
def plot_confusion_matrix(labels, pred_labels, save_path):
    target_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']  

    print(classification_report(labels, pred_labels, target_names=target_names))

    ConfusionMatrix = confusion_matrix(labels, pred_labels)

    print("Confusion matrix on test set for resnet 50b FINAL")
    print(ConfusionMatrix)

    CM = ConfusionMatrix.astype('float') / ConfusionMatrix.sum(axis=1)[:, np.newaxis]
    print("Classwise accuracy on test set of model", CM.diagonal()*100)
    
    mae = mean_absolute_error(labels, pred_labels)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

    mse = mean_squared_error(labels, pred_labels)
    print(f"Mean Squared Error (MSE): {mse:.4f}")

    
    plt.figure(figsize=(10, 8))
    sns.heatmap(ConfusionMatrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix(Res-Net-50B)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.savefig(save_path)
    plt.close()


confusion_matrix_save_path = 'confusion_matrix Resnet-50b FINAL.png'
plot_confusion_matrix(all_labels, all_preds, confusion_matrix_save_path)
print(f"Confusion matrix saved to {confusion_matrix_save_path}")
torch.save(model.state_dict(), "Resnet-50bb FINAL.pth")
