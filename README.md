# Implementation-of-Transfer-Learning

## Aim
To Implement Transfer Learning for classification using VGG-19 architecture.
## Problem Statement and Dataset
Image classification is a core task in computer vision where the objective is to categorize an image into one of several predefined classes. Training deep neural networks from scratch requires large datasets and extensive computational resources. To overcome this, Transfer Learning allows us to use a pre-trained model (such as VGG-19 trained on ImageNet) and fine-tune it for our specific dataset.

In this experiment, the VGG-19 model is used for classifying images into the target dataset categories. The dataset is split into training, validation, and test sets, where the model learns feature representations from the training set and its performance is validated and tested on unseen data.

## DESIGN STEPS
### STEP 1:
Import the necessary libraries such as PyTorch, Torchvision, and Matplotlib.  
</br>

### STEP 2:
Load the dataset and apply preprocessing (resizing, normalization, and augmentation).  
</br>

### STEP 3:
Download the pre-trained VGG-19 model from Torchvision models.  
</br>

### STEP 4:
Freeze the feature extraction layers of VGG-19.  
</br>

### STEP 5:
Modify the final fully connected layer to match the number of dataset classes.  
</br>

### STEP 6:
Define the loss function (CrossEntropyLoss) and optimizer (Adam/SGD).  
</br>

### STEP 7:
Train the model on the training dataset and validate on the validation set.  
</br>

### STEP 8:
Plot Training Loss and Validation Loss vs Iterations.  
</br>

### STEP 9:
Evaluate the model on the test dataset.  
</br>

### STEP 10:
Generate Confusion Matrix, Classification Report, and test on new sample images.  
</br>

## PROGRAM
### Developed By: K MADHAVA REDDY
### Register Number: 212223240064
```python
# Load Pretrained Model and Modify for Transfer Learning
from torchvision.models import VGG19_Weights
model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)


# Modify the final fully connected layer to match the dataset classes
num_ftrs = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(num_ftrs, 1)


# Include the Loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



# Train the model
def train_model(model, train_loader,test_loader,num_epochs=10):
    train_losses = []
    val_losses = []
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels.unsqueeze(1).float()) # Reshape labels and convert to float
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_losses.append(running_loss / len(train_loader))

        # Compute validation loss
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.unsqueeze(1).float()) # Reshape labels and convert to float
                val_loss += loss.item()

        val_losses.append(val_loss / len(test_loader))
        model.train()

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, Validation Loss: {val_losses[-1]:.4f}')

    # Plot training and validation loss
    print("Name: K MADHAVA REDDY")
    print("Register Number: 212223240064")
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()
    return model 


```

## OUTPUT
### Training Loss, Validation Loss Vs Iteration Plot
<img width="838" height="653" alt="image" src="https://github.com/user-attachments/assets/fdb1caa5-8378-4c12-8c9c-818ab67fb2fc" />


### Confusion Matrix
<img width="790" height="683" alt="image" src="https://github.com/user-attachments/assets/cd595ae5-15fd-47b0-83ab-9b27c111267d" />


### Classification Report
<img width="578" height="257" alt="image" src="https://github.com/user-attachments/assets/6713d0bc-ae7f-451e-8927-adf4ad625dd8" />

### New Sample Prediction
<img width="706" height="440" alt="image" src="https://github.com/user-attachments/assets/6222ccaf-e72a-4e50-b82d-d85aed959109" />

<img width="621" height="439" alt="image" src="https://github.com/user-attachments/assets/828da295-775c-4fa6-b217-5352edfff51c" />

## RESULT
Thus, the VGG-19 transfer learning model was successfully implemented for image classification.
