from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image, deprocess_image
from ResAttention import *
import glob
import cv2
import numpy as np
import torchvision.transforms as transforms
import torch
from PIL import Image

model = ResidualAttentionModel_92(num_classes=3, feat_ext=False)
model.load_state_dict(torch.load('./weights/all.pth'))
target_layers = [model.residual_block6]

data_transform = transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# Note: input_tensor can be a batch tensor with several images!

# Construct the CAM object once, and then re-use it on many images:
cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

# You can also use it within a with statement, to make sure it is freed,
# In case you need to re-create it inside an outer loop:
# with GradCAM(model=model, target_layers=target_layers, use_cuda=args.use_cuda) as cam:
#   ...

# We have to specify the target we want to generate
# the Class Activation Maps for.
# If targets is None, the highest scoring category
# will be used for every image in the batch.
# Here we use ClassifierOutputTarget, but you can define your own custom targets
# That are, for example, combinations of categories, or specific outputs in a non standard model.

transform = transforms.Compose([transforms.PILToTensor()])

# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
i = 0
for filename in glob.glob('../images/train_full/*/*.jpg'):
    i += 1
    if i % 10 != 0:
        continue
    if i == 100:
        break
    rgb_img = Image.open(filename)
    #input_tensor = preprocess_image(rgb_img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    input_tensor = data_transform(rgb_img).view(-1, 3, 224, 224)
    print(input_tensor.shape)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None, aug_smooth=True, eigen_smooth=True)
    grayscale_cam = grayscale_cam[0]
    grayscale_cam = 1 - grayscale_cam
    rgb_img = cv2.imread(filename, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    resized = cv2.resize(rgb_img, (224, 224), interpolation = cv2.INTER_AREA)
    visualization = show_cam_on_image(resized, grayscale_cam, use_rgb=True)
    cv2.imwrite('visualization/{}.jpg'.format(i), visualization)
