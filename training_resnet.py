from torchvision.transforms import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
from torch import nn
import torchvision.models as models
from torch import optim
from torch.random import manual_seed
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

resnet50 = models.resnet50(pretrained= True)
resnet50.fc = nn.Linear(resnet50.fc.in_features, 5)

resnet50.to(device)
for name, param in resnet50.named_parameters():
    if name not in ["fc.weight", "fc.bias"]:
        param.requires_grad = False

class RegularizedResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()  # Corrected super() usage
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Sequential(
            nn.Dropout(0.45),  # Add dropout layer
            nn.Linear(self.resnet50.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet50(x)

model = RegularizedResNet50(5)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

optimizer = optim.SGD(model.parameters() , lr = 0.02)
lossfn = nn.CrossEntropyLoss()

data_set =ImageFolder(root=r"C:\Users\AATISH KUMAR\Desktop\work_02\new_data" , transform=transform)
my_classes = data_set.classes
print(my_classes[3])

from torch.random import manual_seed
manual_seed(1)
BATCH_SIZE = 32
batch_data = DataLoader(data_set , batch_size= BATCH_SIZE , shuffle= True )
for batch in batch_data:
  image , label = batch
  


def model_training(model, epochs, optimizer, lossfn, batch_data):
    model.to(device)  # Move the model to the GPU
    manual_seed(1)
    loss_values = []
    epoch_count = []

    for epoch in range(epochs):
        total_loss = 0
        model.train()

        for batch_idx, (image, label) in enumerate(batch_data):
            image, label = image.to(device), label.to(device)  # Move data to the GPU
            
            label_logits = model(image)
            label_pred = torch.argmax(nn.functional.softmax(label_logits , dim = 1), dim=1)
            if batch_idx %17 == 0 :
              print(batch_idx ,label_pred)
              print(label_pred==label)
            loss = lossfn(label_logits, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(batch_data)
        loss_values.append(avg_loss)

        epoch_count.append(epoch)
        print(f'epoch = {epoch} | error = {(avg_loss)*100}')

    # Plot the training loss versus epoch
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_count, loss_values, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Epoch')
    plt.legend()
    plt.show()

    trained_model = model
    return trained_model

if '__name__'=='__main__':
    trained_model = model_training(model ,25,optimizer , lossfn , batch_data)

    torch.save(trained_model, r'C:\Users\AATISH KUMAR\Desktop\work_02\trained_model_resnet_01.pth')
    print('hello')