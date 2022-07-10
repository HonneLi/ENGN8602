# ENGN8602
Deep learning methods have yielded many results in the field of facial expression recognition recently, however, many studies only focused on the recognition of still images, rather than changes in facial expression over a continuous time. We evaluated a structure that can be implemented easily to classify people’s emotional status in a video. We compared various deep learning models in video classification fields, and we implemented a network with residual attention module to extract video features and visualized its effectiveness. We demonstrated that this feature extractor module significantly improves model’s overall performance. This work highlights the benefits of attention mechanisms in analyzing human’s facial expressions across a period of time. We also tried its performance subject-wisely and classification ability of more refined labels.

## Requirement
See [requirement.txt](./requirement.txt) for further details.

## Dataset
[SENDv1](https://github.com/StanfordSocialNeuroscienceLab/SEND) dataset is a set of rich, multimodal videos of self-paced, unscripted emotional narratives, annotated for emotional valence over time. The complex narratives and naturalistic expressions in this dataset provide a challenging test for contemporary time-series emotion recognition models. 
<p  align="middle">
  <img src="./sample/dys.png" width="150" />
  <img src="./sample/koi.png" width="150" />
  <img src="./sample/met.png" width="150" />
  <img src="./sample/par.png" width="150" />
  <img src="./sample/sup.png" width="150" />
  <br>
  Multi-cells images in SIPaKMeD dataset
</p>

In this project, however, we utilized only the visual features to recognize emotional status.


## Model
All models are in the folder `models`, which contains the baseline and all other methods used in this paper. 
- ResNet50 - baseline
- Residual Attention Network - model 1
- DenseNet121 - model 2
- DenseNet121 with Residual Channel Attention - model 3



### Results
Details of results are in [here](./Results.md) for further details.

The residual channel attention mechaism shows its effectiveness in the classification of multi-cells images.


<p  align="middle">
  <img src="./vis_densenet.jpeg" width="550" />
  <br>
  (a) DenseNet-121 without Attention
  <br>
  <br>
  <br>
  <img src="./vis_att_densenet.jpeg" width="550" />
  <br>
  (b) DenseNet-121 with Attention
</p>

## Reference
For the MS-TCN: https://github.com/yabufarha/ms-tcn


For the residual channel attention network: https://github.com/yulunzhang/RCAN
