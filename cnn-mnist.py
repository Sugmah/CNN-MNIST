import torch 
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import random

# Set random seed for reproducibility, we picked 42 as it a great number
torch.manual_seed(42)

# Hyperparameters and Data Loaders, used the numbers from the original sample code
num_epochs = 10
num_classes = 10
batch_size = 256
learning_rate = 0.001

DATA_PATH = 'data/'
MODEL_STORE_PATH = 'models/'

# Transforms to apply to the data
trans = transforms.Compose([transforms.ToTensor()])

# MNIST dataset
train_dataset = datasets.MNIST(root=DATA_PATH, train=True, transform=trans, download=True)
test_dataset = datasets.MNIST(root=DATA_PATH, train=False, transform=trans)

# Data loaders
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=4, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=4, shuffle=False)

# Split the training dataset into train and validation sets
val_size = int(0.2 * len(train_dataset))
train_size = len(train_dataset) - val_size
train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# Define the CNN model
class ImprovedCNN(nn.Module):
    def __init__(self):
        super(ImprovedCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 32 feature maps at the size of 28x28
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),       # Shrinks the feature map to 32 14x14 maps

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 64 feature maps at the size of 14x14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)        # Shrinks the feature map to 64 7x7 maps
        )
        self.classifier = nn.Sequential( # Created the classifier 
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), 
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Instantiate the model, loss function, and optimizer
# used pyTorch and cuda to use the GPU, we have a failsafe in the event no GPU is available. 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ImprovedCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
def train_model(model, train_loader, val_loader, num_epochs):
    train_loss_history, val_loss_history = [], []
    train_acc_history, val_acc_history = [], []

    for epoch in range(num_epochs): 

        # Training phase
        model.train()  # Set the model to training mode
        train_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_accuracy = correct / total

        # Validation phase
        model.eval() # Set the model to use the evaluation mode
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # Compute the average 
        val_loss /= len(val_loader)
        val_accuracy = correct / total

        # Log & Storing of Metrics  
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        train_acc_history.append(train_accuracy)
        val_acc_history.append(val_accuracy)

        # Print the values and metrics for each epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        
    # Return the history of losses and accuracies
    return train_loss_history, val_loss_history, train_acc_history, val_acc_history 

# Train the model
train_loss_history, val_loss_history, train_acc_history, val_acc_history = train_model(
    model, train_loader, val_loader, num_epochs
)

# Evaluate on test set with detailed metrics
def evaluate_detailed_metrics(model, test_loader):
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    # Compute and print classification metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds))

evaluate_detailed_metrics(model, test_loader)

# Visualize training and validation loss/accuracy
plt.figure(figsize=(12, 6))
plt.plot(train_loss_history, label='Train Loss') # Plot the training loss across the epochs
plt.plot(val_loss_history, label='Val Loss') # plot the validation accuracy across the epochs
plt.legend()
plt.title('Loss vs. Epochs')
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(train_acc_history, label='Train Accuracy') # Plot the training loss across the epochs
plt.plot(val_acc_history, label='Val Accuracy') # plot the validation accuracy across the epochs
plt.legend()
plt.title('Accuracy vs. Epochs')
plt.show()

# Function to print random data samples with labels
def print_data_samples(dataset, num_samples=5):
    # Display random samples from the dataset with their labels.
    samples = random.sample(range(len(dataset)), num_samples)
    plt.figure(figsize=(12, 6))
    for idx, sample_idx in enumerate(samples):
        image, label = dataset[sample_idx]
        plt.subplot(1, num_samples, idx + 1)
        plt.imshow(image.squeeze(), cmap='gray')
        plt.title(f"Label: {label}")
        plt.axis('off')
    plt.suptitle("Random Data Samples", fontsize=16)
    plt.show()

# Visualize feature maps for both Layer 1 and Layer 2 with different colormaps
def visualize_feature_maps(model, dataset, layer_indices, num_samples=5):
    # Display feature maps for specified layers and dataset samples.
    model.eval() # Tell the model to continue evaluating
    samples = random.sample(range(len(dataset)), num_samples) # Randomly selects num_samples from the data set and puts it into a visualization map.

    for layer_idx in layer_indices:
        colormap = 'gray' if layer_idx == 0 else 'viridis'  # Grayscale for Layer 1, Colored for Layer 2
        plt.figure(figsize=(15, num_samples * 3))
        for idx, sample_idx in enumerate(samples):
            image, label = dataset[sample_idx]
            image_tensor = image.unsqueeze(0).to(device)  # Add batch dimension
            with torch.no_grad(): # Disable the gradident compute for more effective optimization of the computer
                features = model.features[:layer_idx + 1](image_tensor) # extract the feature maps up to the specific layer from the layer_idx
            num_filters = min(16, features.size(1))

            for i in range(num_filters): # loop through the filters in the feature map 
                plt.subplot(num_samples, num_filters, idx * num_filters + i + 1) # Create a subplot for each of the filters 
                plt.imshow(features[0, i].cpu().numpy(), cmap=colormap) # Display the filter as a 2d image to then apply the appropriate colormap
                if i == 0:
                    plt.ylabel(f'Label: {label}', fontsize=12)
                plt.axis('off') # disable the plot axis for the graphs

        plt.suptitle(f'Feature Maps from Layer {layer_idx + 1}', fontsize=16) # Add a title to the feature maps and label it with their specific layer
        plt.show()

# Print random data samples
print_data_samples(test_dataset, num_samples=10)

# Visualize feature maps from both Layer 1 in a grayscale layout and Layer 2 in a colored layout. 
visualize_feature_maps(model, test_dataset, layer_indices=[0, 1], num_samples=10)
