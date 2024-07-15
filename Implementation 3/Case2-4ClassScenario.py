import os
import cv2
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report, mean_squared_error, mean_absolute_error
import pandas as pd
import torch.optim as optim
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.models as models
from PIL import Image
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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



#combine classes
def Adjust_labels(label):
    if (label == 1 or label == 2):
        label = 12
        return label
    else:
        return label

def remap_labels(label):
    label_map = {0: 0, 12: 1, 3: 2, 4: 3}
    return label_map[label]

def preprocess_data(csv_file):
    df = pd.read_csv(csv_file)
    df['Retinopathy grade'] = df['Retinopathy grade'].apply(Adjust_labels)
    df['Retinopathy grade'] = df['Retinopathy grade'].apply(remap_labels)
    return df



train_data = preprocess_data(train_csv_file)
# print(train_data)

# print("Unique labels after preprocessing:")
# print(train_data['Retinopathy grade'].unique())
# print("Number of unique labels:", len(train_data['Retinopathy grade'].unique()))
# quit()

num_classes = len(set(train_data['Retinopathy grade']))
model = models.vgg19(weights='DEFAULT', progress=True)
model.fc = torch.nn.Linear(model.classifier[6].in_features, num_classes)     
model = model.to(device)

#For ResNet
# model = models.resnet50(weights='DEFAULT', progress=True)
# model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# For AlexNet:
# model = models.alexnet(weights='DEFAULT', progress=True)
# model.fc = torch.nn.Linear(model.classifier[6].in_features, num_classes) 


for param in model.parameters():
    param.requires_grad = False
for param in model.classifier[6].parameters():
    param.requires_grad = True

# Cross-Validation
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
    print("Training Stage-1")
    optimizer_adadelta = optim.Adadelta(model.classifier.parameters(), lr=1.0, rho=0.9)
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
        else:
            pass
        
    model.load_state_dict(best_model_weights)


    print("Training Stage-2")

    optimizer_sgd = optim.SGD(model.classifier.parameters(), lr=0.001)
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
            optimizer_sgd.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer_sgd.step()

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
    torch.save(model.state_dict(), "vgg16_combineclass.pth")

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
average_accuracy = np.mean(fold_accuracies)
print(f"Average Accuracy across all folds: {average_accuracy:.2f}%")   

# Convert to numpy arrays
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_preds_prob = np.array(all_preds_prob)


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
    plt.savefig('multiclass_roc_and_hist_vgg16.png', dpi=300, bbox_inches='tight')
    plt.close()
    return roc_auc

n_classes = 4
class_names = ['Class 0', 'Class 1+2', 'Class 3', 'Class 4']

plot_multiclass_roc_with_hist(all_labels, all_preds_prob, num_classes, class_names)
print("Multiclass ROC curves and histograms saved as 'multiclass_roc_and_hist_vgg16.png'")


#calculations for Confusion matrix, MAE, MSE, classwise accuracy
def plot_confusion_matrix(labels, pred_labels, save_path):
    target_names = ['Class 0', 'Class 1+2', 'Class 3', 'Class 4']  # Adjust based on your number of classes

    print(classification_report(labels, pred_labels, target_names=target_names))

    ConfusionMatrix = confusion_matrix(labels, pred_labels)

    print("Confusion matrix on test set")
    print(ConfusionMatrix)

    CM = ConfusionMatrix.astype('float') / ConfusionMatrix.sum(axis=1)[:, np.newaxis]
    print("Classwise accuracy on test set of model", CM.diagonal()*100)
    
    mae = mean_absolute_error(labels, pred_labels)
    print(f"Mean Absolute Error (MAE): {mae:.4f}")

    mse = mean_squared_error(labels, pred_labels)
    print(f"Mean Squared Error (MSE): {mse:.4f}")

    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(ConfusionMatrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix(VGG-16)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save confusion matrix
    plt.savefig(save_path)
    plt.close()

# Plot and save confusion matrix
confusion_matrix_save_path = 'confusion_matrix_vgg16.png'
plot_confusion_matrix(all_labels, all_preds, confusion_matrix_save_path)
print(f"Confusion matrix saved to {confusion_matrix_save_path}")


torch.save(model.state_dict(), "vgg16_combineclasses.pth")
