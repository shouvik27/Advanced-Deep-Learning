import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

class RotatedMNISTDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.angles = [0, 90, 180, 270]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, _ = self.dataset[index]
        angle = self.angles[index % len(self.angles)]
        rotated_img = transforms.functional.rotate(img, angle)
        angle_label = self.angles.index(angle)
        return rotated_img, angle_label

def get_data_loaders(batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    rotated_train_dataset = RotatedMNISTDataset(train_dataset)
    rotated_test_dataset = RotatedMNISTDataset(test_dataset)

    train_loader = DataLoader(rotated_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(rotated_test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
