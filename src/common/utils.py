from torchvision import transforms

def get_normalization_transform():
    data_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return data_transform
