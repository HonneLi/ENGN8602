from torchvision import transforms, datasets
import torch

def get_dataloaders(input_size, batch_size, shuffle = True):
    '''
    Create the dataloaders for train, validation and test set. Randomly rotate images for data augumentation
    Normalization based on std and mean.
    '''
    data_transform = transforms.Compose([transforms.RandomRotation(25),
                                         transforms.Resize(input_size),
                                         transforms.CenterCrop(input_size),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_dataset = datasets.ImageFolder(root='../images/train_three/', transform=data_transform)
    val_dataset = datasets.ImageFolder(root='../images/train_three/', transform=data_transform)
    dataLoader = {'train': torch.utils.data.DataLoader(train_dataset,
                                             batch_size=batch_size, 
                                             shuffle=True,
                                             num_workers=8), 
                  'valid': torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=8)}
    return dataLoader
