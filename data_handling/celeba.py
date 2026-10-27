from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from data_handling.base import BaseDataModuleClass
from sklearn.model_selection import train_test_split

from project_paths import CELEBA_ROOT, CELEBA_ATTR_PATH as ATTR_PATH, CELEBA_SPLIT_DIR as SPLIT_DIR

class CelebADataset(Dataset):
    def __init__(
        self,
        attr_df: pd.DataFrame,
        target_size: tuple = (256, 256),
        transform: callable = None,
        parents: list = None,
    ):
        """
        Args:
            attr_df: DataFrame with index as filenames and columns as attributes
            target_size: (H, W) target image size
            transform: Optional image transform (if None, defaults to standard)
            parents: List of attributes to include in each sample as `pa`
        """
        super().__init__()
        self.target_size = target_size
        self.transform = transform
        self.parents = parents
        self.attr_df = attr_df
        self.filenames = list(attr_df.index)

        # Sanity check
        if self.parents is not None:
            for p in self.parents:
                if p not in self.attr_df.columns:
                    raise ValueError(f"Attribute '{p}' not in attr_df columns.")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx: int) -> dict:
        fname = self.filenames[idx]
        img_path = CELEBA_ROOT / fname

        # Load and transform image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ])(image)

        sample = {"x": image}

        # Add conditioning attributes
        if self.parents is not None:
            raw_values = self.attr_df.loc[fname]
            parent_tensors = []
            for p in self.parents:
                val = (raw_values[p] + 1) // 2  # -1/1 → 0/1
                t = torch.tensor(val).unsqueeze(-1).float()
                sample[p] = t
                parent_tensors.append(t)

            sample["pa"] = torch.cat(parent_tensors).detach()

        return sample

    def get_class_array_for_sampling(self, attrs: list) -> np.ndarray:
        """
        Args:
            attrs: List of attributes to balance over (subset of attr_df columns)

        Returns:
            class_array: np.ndarray of shape (N,) with encoded composite class labels
        """
        if not all(a in self.attr_df.columns for a in attrs):
            raise ValueError("One or more balancing attributes not found in attr_df.")

        attr_vals = (self.attr_df[attrs].values.astype(np.int64) + 1) // 2
        _, composite_labels = np.unique(attr_vals, axis=0, return_inverse=True)
        return composite_labels



class CelebADataModule(BaseDataModuleClass):
    def create_datasets(self):
        self.target_size = (self.args.image_size, self.args.image_size)
        self.parents = self.args.parents

        # Load and preprocess attributes
        attr_df = pd.read_csv(ATTR_PATH, sep=r"\s+", skiprows=1)
        attr_df.index = [f"{i}.jpg" for i in range(len(attr_df))]

        # CSV paths
        train_csv = SPLIT_DIR / "celeba_train.csv"
        val_csv = SPLIT_DIR / "celeba_val.csv"
        test_csv = SPLIT_DIR / "celeba_test.csv"

        # Create or load splits
        if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
            SPLIT_DIR.mkdir(parents=True, exist_ok=True)
            print("🔧 Creating new split and saving to CSV...")
            filenames = attr_df.index.values
            train_val_idx, test_idx = train_test_split(filenames, test_size=0.1, random_state=42)
            train_idx, val_idx = train_test_split(train_val_idx, test_size=0.1111, random_state=42)  # 0.1111 * 0.9 ~= 0.1

            attr_df.loc[train_idx].to_csv(train_csv)
            attr_df.loc[val_idx].to_csv(val_csv)
            attr_df.loc[test_idx].to_csv(test_csv)
        else:
            print("✅ Found existing CSV splits. Loading...")

        df_train = pd.read_csv(train_csv, index_col=0)
        df_val = pd.read_csv(val_csv, index_col=0)
        df_test = pd.read_csv(test_csv, index_col=0)

        print(f"Train size: {len(df_train)}, Val size: {len(df_val)}, Test size: {len(df_test)}")

        self.dataset_train = CelebADataset(
            attr_df=df_train,
            target_size=self.target_size,
            transform=None,
            parents=self.parents,
        )

        self.dataset_val = CelebADataset(
            attr_df=df_val,
            target_size=self.target_size,
            transform=None,
            parents=self.parents,
        )

        self.dataset_test = CelebADataset(
            attr_df=df_test,
            target_size=self.target_size,
            transform=None,
            parents=self.parents,
        )

    @property
    def dataset_name(self):
        return "celeba"
