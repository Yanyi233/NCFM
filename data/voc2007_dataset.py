import torch
from torch.utils.data import Dataset
import os

class VOC2007_PT_Dataset(Dataset):
    """
    Loads pre-processed VOC2007 data from a .pt file.
    The .pt file is expected to contain dictionaries with keys like:
    'classes': list of class names
    'images_train', 'labels_train': training images (N, C, H, W) and labels (N, num_classes)
    'images_test', 'labels_test': testing images and labels
    """
    def __init__(self, pt_file_path, train=True, transform=None, target_transform=None):
        if not os.path.exists(pt_file_path):
            raise FileNotFoundError(f"Error: Data file not found {pt_file_path}")

        print(f"Loading VOC2007 data from {pt_file_path}...")
        try:
            # map_location='cpu' to avoid loading directly onto GPU potentially causing OOM
            self.data_dict = torch.load(pt_file_path, map_location='cpu')
        except Exception as e:
            raise RuntimeError(f"Error loading {pt_file_path}: {e}")

        self.train = train
        self.transform = transform
        self.target_transform = target_transform

        # Extract data based on train/test split
        image_key = 'images_train' if self.train else 'images_test'
        label_key = 'labels_train' if self.train else 'labels_test'

        if image_key not in self.data_dict or label_key not in self.data_dict:
            raise KeyError(f"Error: Missing keys '{image_key}' or '{label_key}' in {pt_file_path}")

        self.images = self.data_dict[image_key]
        self.labels = self.data_dict[label_key]
        self.classes = self.data_dict.get('classes', None) # Optional: get class names if available

        # Ensure labels are FloatTensor for BCEWithLogitsLoss
        if self.labels.dtype != torch.float32:
            self.labels = self.labels.float()

        print(f"Loaded {len(self.images)} {'training' if self.train else 'test'} samples.")
        if self.classes:
            print(f"Dataset classes: {self.classes}")
            # Optional: Add check for num_classes consistency here if needed

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img, target = self.images[index], self.labels[index]

        # Apply transformations if any (though images in .pt might already be transformed)
        # Note: Standard torchvision transforms might not be suitable if data is already normalized tensor
        if self.transform:
            # This assumes self.transform expects a Tensor, adjust if it expects PIL Image
            img = self.transform(img)

        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def get_classes(self):
        return self.classes

    def get_num_classes(self):
        return self.labels.shape[1] # Get num_classes from label dimension 