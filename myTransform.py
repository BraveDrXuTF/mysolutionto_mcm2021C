import torchvision.transforms as transforms
def myTransform(x):
    x=transforms.RandomHorizontalFlip(p=0.5)(x)
    x=transforms.RandomVerticalFlip(p=0.5)(x)
    x=transforms.ToTensor()(x)
    return(x)
    