import torch
from torchvision import datasets
import torch.nn as nn
import torch.optim as optim
import os 
import glob

from ResAttention import *

from train_evaluate import extract
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--num_classes", help="number of predicted classes",
                    type=int, default="9")
parser.add_argument("--video_dir", help="video directory",
                    type=str, default="../face_video/valid/raw/")
parser.add_argument("--save_dir", help="saving directory of extracted features",
                    type=str, default="./valid/full/")
parser.add_argument("--weight_dir", help="pretrained weight directory",
                    type=str, default="weights/fullset.pth")
args = parser.parse_args()

num_classes = 9

model = ResidualAttentionModel_92(num_classes=args.num_classes, feat_ext=True)
#model = model.load_state_dict(torch.load(args.weight_dir))
model.load_state_dict(torch.load(args.weight_dir))
model = model.to(device)

print("Extraction progress")
print("=" * 20)
set_list = os.listdir(args.video_dir)
for path in set_list:
    vids = []
    root = args.video_dir + path + '/*.mp4'
    for filename in glob.glob(root):
        vids.append(filename)
    save_dir = args.save_dir + path
    os.makedirs(save_dir, exist_ok=True)
    extract(model=model, dataloader=vids, save_dir=save_dir)
