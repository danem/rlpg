import torch.utils.data as tud
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
from PIL import Image

class ImageDataset(Dataset):
  def __init__(self, root_dir, transform=None):
    self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir) if img.endswith(".png")]
    self.transform = transform

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    image_path = self.image_paths[idx]
    image = self.load_image(image_path)
    if self.transform:
      image = self.transform(image)
    return image

  def load_image(self, image_path):
    return Image.open(image_path)
  

def image_dataloader (path, img_transforms = None):
    img_transforms = img_transforms if img_transforms else []

    transform = transforms.Compose([transforms.ToTensor()] + img_transforms)
    dataset = ImageDataset(path, transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=6)
    return dataloader

def split_dataset (ds, train_pct):
    total_len = len(ds)
    train_len = int(train_pct * total_len)
    val_len = total_len - train_len
    return tud.random_split(ds, [train_len, val_len])