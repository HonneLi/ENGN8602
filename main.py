import torch
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim
import os 
import numpy as np

from ResAttention import *

from dataLoader import get_dataloaders
from train_evaluate import train_model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = 3
batch_size = 16
shuffle_datasets = True
num_epochs = 100
save_dir = "weights"
os.makedirs(save_dir, exist_ok=True)
save_all_epochs = True

model = ResidualAttentionModel_92(num_classes = num_classes)
#model = model.load_state_dict(torch.load('weights/fullset.pth'))

model = model.to(device)
dataLoader = get_dataloaders(input_size=224, batch_size=batch_size, shuffle=shuffle_datasets)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# training
print("Training progress")
print("=" * 20)
trained_model, train_losses, train_acc, val_losses, val_acc = train_model(model=model, dataloaders=dataLoader, criterion=criterion, optimizer=optimizer, save_dir=save_dir, save_all_epochs=save_all_epochs, num_epochs=num_epochs)
#vids = []
#for filename in glob.glob('../train/Positive/*.mp4'):
#    vids.append(filename)
#res = evaluate(model=model, dataloaders=vids)
# save the model
torch.save(trained_model.state_dict(), "weights/three_100.pth")

# plot loss and accuracy
"""
print()
print("Plots of loss and accuracy during training")
print("=" * 20)

x = np.arange(0,50,1)
plt.plot(x, train_losses, label='Training loss')
plt.plot(x, val_losses, label='Validation loss')
plt.legend(frameon=False)
plt.title("Pre-trained Resnet50")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

plt.plot(x, train_acc, label='Training accuracy')
plt.plot(x, val_acc, label='Validation accuracy')
plt.legend(frameon=False)
plt.title("Pre-trained Resnet50")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.show()
"""
