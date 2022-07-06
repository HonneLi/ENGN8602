import torch
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim
import os 
import glob

from ResAttention import *

from dataLoader import get_dataloaders
from train_evaluate import extract
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

num_classes = 9
batch_size = 16
shuffle_datasets = True
num_epochs = 50
save_all_epochs = True

model = ResidualAttentionModel_92(num_classes=num_classes, feat_ext=True)
#model = model.load_state_dict(torch.load('weights/fullset.pth'))
model.load_state_dict(torch.load('./weights/nine_raw.pth'))
model = model.to(device)

print("Extraction progress")
print("=" * 20)
set_list = os.listdir('../face_video/valid/raw/')
for path in set_list:
    vids = []
    root = '../face_video/valid/raw/' + path + '/*.mp4'
    for filename in glob.glob(root):
        vids.append(filename)
    save_dir = './valid/full/' + path
    os.makedirs(save_dir, exist_ok=True)
    extract(model=model, dataloader=vids, save_dir=save_dir)
