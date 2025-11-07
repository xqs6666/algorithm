import os,sys
from torchvision import datasets,transforms

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

train_dataset = datasets.ImageFolder(
    root=os.path.join(os.path.dirname(os.path.abspath(__file__)),"data/train"),
    transform=train_transform
)

print(train_dataset.class_to_idx)