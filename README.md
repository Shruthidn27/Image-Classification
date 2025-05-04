# Convolutional Deep Neural Network for Image Classification

## AIM

To Develop a convolutional deep neural network for image classification and to verify the response for new images.

## Problem Statement and Dataset:

The problem statement focuses on developing a Convolutional Neural Network (CNN) for image classification. The goal is to classify images into predefined categories based on learned patterns.

The dataset typically consists of labeled images belonging to distinct classes. In this case, the example suggests classifying handwritten digits (0-9), which aligns with popular datasets like MNIST. Each image in the dataset represents a digit, with corresponding labels allowing the model to learn and predict digit classes.



## Neural Network Model


![image](https://github.com/user-attachments/assets/1651a2c1-2591-4ad2-8864-93892de44f8f)

<br>

## DESIGN STEPS

### STEP 1:
Define the objective of classifying handwritten digits (0-9) using a Convolutional Neural Network (CNN).

### STEP 2:
Convert images to tensors, normalize pixel values, and create DataLoaders for batch processing.

### STEP 3:
Design a CNN with convolutional layers, activation functions, pooling layers, and fully connected layers.

### STEP 4:
Train the model using a suitable loss function (CrossEntropyLoss) and optimizer (Adam) for multiple epochs.

### STEP 5:
Test the model on unseen data, compute accuracy, and analyze results using a confusion matrix and classification report.

### STEP 6:
Save the trained model, visualize predictions, and integrate it into an application if needed.



## PROGRAM

### Name: SHRUTHI D.N
### Register Number: 212223240155
```python
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        # write your code here
        self.conv1=nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2=nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool=nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1=nn.Linear(128*3*3,128)
        self.fc2=nn.Linear(128,64)
        self.fc3=nn.Linear(64,10)

    def forward(self, x):
        # write your code here
        x=self.pool(torch.relu(self.conv1(x)))
        x=self.pool(torch.relu(self.conv2(x)))
        x=self.pool(torch.relu(self.conv3(x)))
        x=x.view(x.size(0),-1)
        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=self.fc3(x)
        return x

```

```python
# Initialize the Model, Loss Function, and Optimizer
model =CNNClassifier()
criterion =nn.CrossEntropyLoss()
optimizer =optim.Adam(model.parameters(),lr=0.001)
```

```python
# Train the Model
def train_model(model, train_loader, num_epochs=3):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')
    print('Name: SHRUTHI D.N ')
    print('Register Number: 212223240155')
        
```

## OUTPUT
### Training Loss per Epoch


![image](https://github.com/user-attachments/assets/0c1c49e3-2dc3-41f6-81f7-08d281715068)

<br>

### Confusion Matrix


![image](https://github.com/user-attachments/assets/6257d7e1-c693-40f8-83dd-5f367c1a9b65)

<br>

### Classification Report


![image](https://github.com/user-attachments/assets/705c9943-1acd-4fed-a9a3-e87d9c4cf5b6)

<br>


### New Sample Data Prediction


![image](https://github.com/user-attachments/assets/f7c4f157-a0fe-4756-8b9a-ae95c62e4dc4)


## RESULT
Thus, a convolutional deep neural network was developed for image classification and verified with the response for new images.
