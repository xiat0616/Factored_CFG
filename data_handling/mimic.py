from pathlib import Path
from typing import Callable, Dict
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
import torchvision
from torchvision.transforms import ToTensor, Resize, CenterCrop
import torch.nn.functional as F
from tqdm import tqdm
from data_handling.base import BaseDataModuleClass
import os

# Load CSV root of MIMIC
from project_paths import MIMIC_ROOT, MIMIC_META

class MimicDataModule(BaseDataModuleClass):
    def create_datasets(self):
        self.target_size = (self.args.image_size, self.args.image_size)
        self.parents = self.args.parents
        print("Train df: ")
        self.dataset_train = MIMICDataset(
            csv_file=MIMIC_META / Path("mimic.sample.train.csv"),
            target_size=self.target_size,
            transform=None,
            parents=self.parents,
        )
        print("Validation df: ")
        self.dataset_val = MIMICDataset(
            csv_file=MIMIC_META / Path("mimic.sample.val.csv"),
            target_size=self.target_size,
            transform=None,
            parents=self.parents,
        )
        print("Test df: ")
        self.dataset_test = MIMICDataset(
            csv_file=MIMIC_META / Path("mimic.sample.test.csv"),
            target_size=self.target_size,
            transform=None,
            parents=self.parents,
        )

    @property
    def dataset_name(self):
        return "mimic"


class MIMICDataset(Dataset):
    def __init__(
        self,
        csv_file: Path,
        target_size: tuple,
        transform: Callable = None,
        parents=None,
    ):
        """
        MimicDataset class for loading and preprocessing MIMIC-CXR data.

        :param csv_file: Path to the CSV file containing metadata.
        :param target_size: Target size for resizing the images.
        :param transform: Transformations to apply to the images.
        :param parents: Parent attributes for causal models.

        """
        super().__init__()

        # Load and preprocess the CSV file
        df = pd.read_csv(csv_file)
        df = df[df['disease'] != 'Other']  # Focus only on Pleural Effusion
        print(f"Sex distribution:\n{df.sex.value_counts(normalize=True)}")
        print(f"Disease distribution:\n{df.disease.value_counts(normalize=True)}")
        print(f"Race distribution:\n{df.race.value_counts(normalize=True)}")
        df['disease'] = df['disease'].replace({'No Finding': 0, 'Pleural Effusion': 1})
        print(f"Dataset size before sharding: {len(df)}")
        df.fillna(0, inplace=True)
 

        # Store dataset attributes
        self.img_paths = df.path_preproc.values
        self.sex = df.sex_label.values
        self.age = df.age.values
        self.race = df.race_label.values
        self.finding = df.disease.values
        self.transform = transform
        self.parents = parents
        self.target_size = target_size

    def __len__(self) -> int:
        """
        Returns the length of the dataset.
        """
        return len(self.img_paths)

    def read_image(self, idx: int) -> torch.Tensor:
        """
        Reads and preprocesses an image from disk.

        :param idx: Index of the image to read.
        :return: Preprocessed image as a tensor.
        """
        img_path = MIMIC_ROOT / "data" / self.img_paths[idx]
        img = io.imread(img_path, as_gray=True)
        img = img / (img.max() + 1e-12)  # Normalize
        img = CenterCrop(self.target_size)(Resize(self.target_size, antialias=True)(ToTensor()(img)))
        return img * 2 -1 # Need to be (-1,1)

    def __getitem__(self, idx: int) -> Dict:
        """
        Retrieves a data sample at a given index.

        :param idx: Index of the sample to retrieve.
        :return: A dictionary containing the sample data.
        """
        img = self.read_image(idx)

        sample = {
            "age": torch.tensor(self.age[idx]).unsqueeze(-1).float() / 100.0,
            "race": F.one_hot(torch.tensor(self.race[idx]), num_classes=3).float(),
            "sex": torch.tensor(self.sex[idx]).unsqueeze(-1).float(),
            "finding": torch.tensor(self.finding[idx]).unsqueeze(-1).float(),
            "x": self.transform(img).float() if self.transform is not None else img.float(),
        }

        # Add parent attributes if specified
        if self.parents is not None:
            sample["pa"] = torch.cat(
                [
                    sample[c]
                    if isinstance(sample[c], torch.Tensor)
                    else torch.tensor([sample[c]])
                    for c in self.parents
                ]
            ).detach()
        sample['short_path'] = os.path.splitext(os.path.basename(self.img_paths[idx]))[0] 
        return sample
