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
        #label = Adjust_labels(label)
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


def preprocess_data(csv_file):
    df = pd.read_csv(csv_file)
    df = df[df['Retinopathy grade'].isin([0, 4])]
    df['Retinopathy grade'] = df['Retinopathy grade'].map({0: 0, 4: 1})
    return df

train_data = preprocess_data(train_csv_file)

num_classes = 2
model = models.alexnet(weights='DEFAULT', progress=True)
model.classifier[6] = torch.nn.Linear(model.classifier[6].in_features, num_classes)     
model = model.to(device)

#freeze all layers
for param in model.parameters():
    param.requires_grad = False
#unfreeze last fc layer
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
    optimizer_adadelta = optim.Adadelta(model.classifier[6].parameters(), lr=1.0, rho=0.9)
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
    

    optimizer_sgd = optim.SGD(model.classifier[6].parameters(), lr=0.001)
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
    torch.save(model.state_dict(), "alexnet_binary.pth")

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
# for i in range(len(all_labels)):
#     print(f"True Label: {all_labels[i]}, Predicted Probabilities: {all_preds_prob[i]}")




def plot_separate_binary_roc_curves(y_true, y_pred_proba):
    class_names = ['Class 0', 'Class 4']
    colors = ['blue', 'red']
    auc_values = []
    
    for i in range(2):  # 0 and 1 (originally 0 and 4)
        if i == 0:
            # For class 0, use 1 - probabilities of class 1
            fpr, tpr, _ = roc_curve(y_true == i, 1 - y_pred_proba[:, 1])
        else:
            # For class 1, use probabilities of class 1
            fpr, tpr, _ = roc_curve(y_true == i, y_pred_proba[:, 1])
        
        roc_auc = auc(fpr, tpr)
        auc_values.append(roc_auc)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color=colors[i], lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic (ROC) Curve for {class_names[i]}')
        plt.legend(loc="lower right")
        
        plt.savefig(f'Aalexnet binary_roc_curve_{class_names[i]}.png', dpi=300, bbox_inches='tight')
        print(f"AUC for {class_names[i]}: {roc_auc:.2f}")
        plt.close()
    
    return auc_values

auc_values = plot_separate_binary_roc_curves(all_labels, all_preds_prob)
print(f"AUC values: {auc_values}")



#calculations for Confusion matrix, MAE, MSE, classwise accuracy
def plot_confusion_matrix(labels, pred_labels, save_path):
    target_names = ['Class 0', 'Class 4']  # Adjust based on your number of classes

    print(classification_report(labels, pred_labels, target_names=target_names))

    ConfusionMatrix = confusion_matrix(labels, pred_labels)

    print("Confusion matrix on test set")
    print(ConfusionMatrix)

    CM = ConfusionMatrix.astype('float') / ConfusionMatrix.sum(axis=1)[:, np.newaxis]
    print("Classwise accuracy on test set of model", CM.diagonal()*100)
    

    n_classes = len(target_names)
    sensitivity = np.zeros(n_classes)
    specificity = np.zeros(n_classes)

    for i in range(n_classes):
        true_positive = ConfusionMatrix[i, i]
        false_negative = np.sum(ConfusionMatrix[i, :]) - true_positive
        false_positive = np.sum(ConfusionMatrix[:, i]) - true_positive
        true_negative = np.sum(ConfusionMatrix) - true_positive - false_negative - false_positive

        sensitivity[i] = true_positive / (true_positive + false_negative)
        specificity[i] = true_negative / (true_negative + false_positive)

    print("\nSensitivity for each class:")
    for i, class_name in enumerate(target_names):
        print(f"{class_name}: {sensitivity[i]:.4f}")

    print("\nSpecificity for each class:")
    for i, class_name in enumerate(target_names):
        print(f"{class_name}: {specificity[i]:.4f}")


    # Plot and save confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(ConfusionMatrix, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix(alexnet)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Save confusion matrix
    plt.savefig(save_path)
    plt.close()

# Plot and save confusion matrix
confusion_matrix_save_path = 'confusion_matrix_alexnet.png'
plot_confusion_matrix(all_labels, all_preds, confusion_matrix_save_path)
print(f"Confusion matrix for binary classes saved to {confusion_matrix_save_path}")


torch.save(model.state_dict(), "alexnet_binaryy.pth")
