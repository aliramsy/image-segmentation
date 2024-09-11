import os
from PIL import Image
import torch
from torchvision import transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, random_split
from model import Unet


mask_dir = './masks/'
image_dir = './images'


def get_mask_unique_pixel_values(mask_dir):
    transform = T.ToTensor()

    unique_values = set()

    for filename in os.listdir(mask_dir):
        if filename.endswith('.png'):
            mask_path = os.path.join(mask_dir, filename)
            msk = Image.open(mask_path).convert('L')
            msk_tensor = transform(msk)
            unique_pixels = torch.unique(msk_tensor)
            unique_values.update(unique_pixels.tolist())

    return torch.tensor(sorted(list(unique_values)))


grayscale_values = get_mask_unique_pixel_values(mask_dir)


def preprocess_mask_distance(mask):
    grayscale_values = get_mask_unique_pixel_values(mask_dir)
    mask[mask == 0.0] = -1

    grayscale_values = grayscale_values[1:]

    distances = torch.abs(mask - grayscale_values.unsqueeze(1).unsqueeze(2))

    # Set ignored values (background) back to 0
    mask[mask == -1] = 0

    return distances


def preprocess_mask_one_hot(mask):
    grayscale_values = get_mask_unique_pixel_values(mask_dir)
    grayscale_values = grayscale_values[1:]

    one_hot_mask = torch.zeros((6, mask.shape[1], mask.shape[2]))
    # Replace 0.0 with -1 (ignore label)
    # mask[mask == 0.0] = -1

    # Calculate the closest class for each pixel
    distances = torch.abs(mask - grayscale_values.unsqueeze(1).unsqueeze(2))
    mask_class = torch.argmin(distances, dim=0)  # Get index of closest class

    # Set ignored values (background) back to -1
    # mask[mask == -1] = 0

    for i in range(6):
        one_hot_mask[i][mask_class == i] = 1

    # return mask_class
    return one_hot_mask


class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None, one_hot=False):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.one_hot = one_hot

        # List of image and mask filenames
        # List of image filenames sorted
        self.images = sorted(os.listdir(image_dir))
        # List of mask filenames sorted
        self.masks = sorted(os.listdir(mask_dir))

        # Extract the common identifier (e.g., 'XX_YYY') from filenames
        self.image_ids = [fname.split('.')[0].split(
            'image')[1] for fname in self.images]
        self.mask_ids = [fname.split('.')[0].split('mask')[1]
                         for fname in self.masks]

        assert self.image_ids == self.mask_ids, "Mismatch between images and masks!"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]

        # Construct file paths for image and corresponding mask
        img_name = f"image{img_id}.jpg"
        mask_name = f"mask{img_id}.png"

        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert(
            "L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            # Preprocess mask to class indices
            if self.one_hot:
                mask = preprocess_mask_one_hot(mask)
            else:
                mask = preprocess_mask_distance(mask)

        return image, mask


def get_train_test_data(one_hot=False):
    torch.manual_seed(42)

    transform = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor()
    ])

    train_dataset = SegmentationDataset(
        image_dir, mask_dir, transform=transform, one_hot=one_hot)

    dataset_size = len(train_dataset)

    train_size = int(0.9 * dataset_size)
    test_size = dataset_size - train_size

    train_dataset, test_dataset = random_split(
        train_dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

    return train_loader, test_loader


if __name__ == '__main__':
    unique_pixels = get_mask_unique_pixel_values(mask_dir)
    print(unique_pixels)

    train_loader, test_loader = get_train_test_data(one_hot=False)
    model = Unet(in_channels=3, out_channels=6)
    for images, labels in train_loader:
        outputs = model(images)
        print(outputs.shape)
        break
