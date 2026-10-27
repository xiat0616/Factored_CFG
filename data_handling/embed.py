from pathlib import Path
from typing import Callable, Dict
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
import torchvision.transforms as T
from torchvision.transforms import ToTensor, Resize, CenterCrop
import torch.nn.functional as F
import os
import cv2
from data_handling.base import BaseDataModuleClass
from sklearn.model_selection import train_test_split
from types import SimpleNamespace

from project_paths import EMBED_ROOT, EMBED_META
import numpy as np

domain_maps = {
    "HOLOGIC, Inc.": 0,
    "GE MEDICAL SYSTEMS": 1,
    "FUJIFILM Corporation": 2,
    "GE HEALTHCARE": 3,
    "Lorad, A Hologic Company": 4,
}

tissue_maps = {
    "A": 0,
    "B": 1, 
    "C": 2, 
    "D": 3
}

modelname_map = {
    "Selenia Dimensions": 0,
    "Senographe Essential VERSION ADS_53.40": 5,
    "Senographe Essential VERSION ADS_54.10": 5,
    "Senograph 2000D ADS_17.4.5": 2,
    "Senograph 2000D ADS_17.5": 2,
    "Lorad Selenia": 3,
    "Clearview CSm": 4,
    "Senographe Pristina": 1,
}

def split_and_save_csvs(
    full_csv_path,
    save_dir="embed_meta",
    label="tissueden",
    domain=None,
    model=None,
    implant_csv_path=None,
):
    image_dir = EMBED_ROOT / Path("images/png/1024x768")
    full_df = pd.read_csv(full_csv_path)

    # Step 1: Remove subjects with implant/device/compression
    if implant_csv_path is not None:
        implant_df = pd.read_csv(implant_csv_path)

        # Normalize image_path for merge
        implant_df["image_path"] = implant_df["image_path"].apply(
            lambda p: "/".join(Path(p).parts[-2:])
        )

        # Identify subjects to remove
        bad_paths = implant_df[
            (implant_df["breast implant"] == 1)
            # | (implant_df["devices"] == 1)
            # | (implant_df["compression"] == 1)
        ]["image_path"].astype(str)

        full_df["image_path_str"] = full_df["image_path"].astype(str)
        bad_subjects = full_df[full_df["image_path_str"].isin(bad_paths)]["empi_anon"].unique()
        print(f"❗ Removing {len(bad_subjects)} subjects with implant/device/compression issues")
        full_df = full_df[~full_df["empi_anon"].isin(bad_subjects)]

        # Extract marker labels
        marker_df = implant_df[["image_path", "circle marker", "triangle marker"]].copy()
        marker_df.columns = ["shortimgpath", "circle", "triangle"]
    else:
        marker_df = pd.DataFrame(columns=["shortimgpath", "circle", "triangle"])

    # Step 2: Preprocessing and merging
    full_df["shortimgpath"] = full_df["image_path"]
    full_df["image_path"] = full_df["image_path"].apply(lambda x: image_dir / str(x))

    if not marker_df.empty:
        full_df = full_df.merge(marker_df, on="shortimgpath", how="left")
    else:
        full_df["circle"] = np.nan
        full_df["triangle"] = np.nan

    # ✅ Keep only samples with valid circle and triangle markers
    full_df = full_df[full_df["circle"].notna() & full_df["triangle"].notna()]
    full_df["circle"] = full_df["circle"].astype(int)
    full_df["triangle"] = full_df["triangle"].astype(int)

    # Manufacturer/domain/model/density processing
    full_df["manufacturer_domain"] = full_df.Manufacturer.apply(lambda x: domain_maps[x])
    full_df["tissueden"] = full_df.tissueden.apply(lambda x: tissue_maps[x])
    full_df["SimpleModelLabel"] = full_df.ManufacturerModelName.apply(lambda x: modelname_map[x])
    full_df["ViewLabel"] = full_df.ViewPosition.apply(lambda x: 0 if x == "MLO" else 1)

    print(f"✅ Total samples before filtering: {len(full_df)}")
    print(f"Triange: {full_df.triangle.value_counts()}")
    print(f"Circle: {full_df.circle.value_counts()}")
    # Final filtering
    full_df = full_df.dropna(
        subset=["age_at_study", "tissueden", "SimpleModelLabel", "ViewLabel", "image_path"]
    )

    df = full_df.copy()
    df = df[df["FinalImageType"] == "2D"]
    df = df[df["tissueden"].notna()]
    df = df[df["tissueden"] < 5]
    df = df[df["ViewPosition"].isin(["MLO", "CC"])]

    print(f"✅ Total samples after filtering: {len(df)}")

    # Step 3: Train/val/test split by patient
    if label == "tissueden":
        y = (
            df.groupby("empi_anon")["tissueden"]
            .unique()
            .apply(lambda x: x[0])
            .values
        )
        train_id, val_id = train_test_split(
            df.empi_anon.unique(), test_size=0.25, random_state=33, stratify=y
        )
    else:
        train_id, val_id = train_test_split(
            df.empi_anon.unique(), test_size=0.25, random_state=33
        )

    test_id = val_id[600:]
    val_id = val_id[:600]

    print(f"📊 N patients: train={len(train_id)}, val={len(val_id)}, test={len(test_id)}")

    # Optional domain/model filtering
    if domain != "None" and domain is not None:
        df = df[df.manufacturer_domain == domain]
        if domain == domain_maps["HOLOGIC, Inc."]:
            df = df[df.ManufacturerModelName == "Selenia Dimensions"]
    if model != "None" and model is not None:
        df = df[df.SimpleModelLabel == model]

    # Print stats
    print("train n images " + str(len(df[df.empi_anon.isin(train_id)])))
    print("test " + str(len(df[df.empi_anon.isin(test_id)])))
    print("val " + str(len(df[df.empi_anon.isin(val_id)])))
    
    # Step 4: Save to CSV
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    df_train = df[df.empi_anon.isin(train_id)]
    df_val = df[df.empi_anon.isin(val_id)]
    df_test = df[df.empi_anon.isin(test_id)]

    if implant_csv_path is not None and "manual" in str(implant_csv_path):
        train_csv = save_dir / "embed_train_manual.csv"
        val_csv = save_dir / "embed_val_manual.csv"
        test_csv = save_dir / "embed_test_manual.csv"
    else:
        train_csv = save_dir / "embed_train.csv"
        val_csv = save_dir / "embed_val.csv"
        test_csv = save_dir / "embed_test.csv"
    df_train.to_csv(train_csv, index=False)
    df_val.to_csv(val_csv, index=False)
    df_test.to_csv(test_csv, index=False)

    print(f"✅ CSVs saved to '{save_dir}':")
    print(f"  Train: {len(df_train)} rows → {train_csv}")
    print(f"  Val:   {len(df_val)} rows → {val_csv}")
    print(f"  Test:  {len(df_test)} rows → {test_csv}")

    return train_csv, val_csv, test_csv

class EmbedDataset(Dataset):
    def __init__(
        self,
        csv_file: Path,
        target_size: tuple,
        transform: Callable = None,
        parents=None,
        binary_density: bool = True,  
    ):
        super().__init__()

        df = pd.read_csv(csv_file)
        print(f"Len df: {len(df)}")
        self.img_paths = df.image_path.values
        self.shortpaths = df.shortimgpath.values
        self.binary_density = binary_density
        # print(f"Raw density distribution:\n{df.tissueden.value_counts(normalize=True)}")

        self.transform = transform
        self.target_size = target_size
        self.scanner = df.SimpleModelLabel.values
        self.age = df.age_at_study.values
        # Triangle, Circle stuff
        self.circle = df.circle.values
        self.triangle = df.triangle.values

        self.parents = parents

        print(f"triangle distribution {df.triangle.value_counts(normalize=True)}")
        print(f"triangle distribution {df.triangle.value_counts(normalize=False)}")

        print(f"circle distribution {df.circle.value_counts(normalize=True)}")
        print(f"circle distribution {df.circle.value_counts(normalize=False)}")

        if self.binary_density:
            # Binarize directly in df: 0 = A/B, 1 = C/D
            self.num_density_classes = 2
            df["tissueden"] = df["tissueden"].apply(lambda x: 0 if x in [0, 1] else 1)
            print(f" Binary density distribution (0=A/B, 1=C/D):\n{df.tissueden.value_counts(normalize=True)}")
        else: 
            self.num_density_classes = 4
        self.density = df.tissueden.values

    def __len__(self):
        return len(self.img_paths)

    def preprocess_image(self, img_path: Path) -> torch.Tensor:
        image = io.imread(img_path, as_gray=True)
        image = image / (image.max() + 1e-12)
        thresh = cv2.threshold((image * 255).astype('uint8'), 5, 255, cv2.THRESH_BINARY)[1]
        nb_components, output, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=4)
        if nb_components > 1:
            max_label = 1 + stats[1:, cv2.CC_STAT_AREA].argmax()
            mask = output == max_label
            image[~mask] = 0
        image = ToTensor()(image)
        image = Resize(self.target_size, antialias=True)(image)
        return image * 2 - 1

    def __getitem__(self, idx: int) -> Dict:
        img = self.preprocess_image(self.img_paths[idx])
        sample = {
            "x": self.transform(img).float() if self.transform is not None else img.float(),
        }

        if "age" in self.parents:
            sample["age"]= torch.tensor(self.age[idx]).unsqueeze(-1).float() / 100.0
        if "density" in self.parents:
            sample["density"]= torch.tensor(self.density[idx]).unsqueeze(-1).float()
        if "scanner" in self.parents:
            sample["scanner"]= F.one_hot(torch.tensor(self.scanner[idx]), num_classes=5).float(),
        if "triangle" in self.parents:
            sample["triangle"] = torch.tensor(self.triangle[idx]).unsqueeze(-1).float()
        if "circle" in self.parents:
            sample["circle"] = torch.tensor(self.circle[idx]).unsqueeze(-1).float()
        if self.parents is not None:
            sample["pa"] = torch.cat([
                sample[c] if isinstance(sample[c], torch.Tensor) else torch.tensor([sample[c]])
                for c in self.parents
            ]).detach()

        return sample


class EmbedDataModule(BaseDataModuleClass):
    def create_datasets(self):
        self.target_size = (self.args.image_size, self.args.image_size)
        self.parents = self.args.parents

        # === Construct datasets ===
        print("Train df: ")
        self.dataset_train = EmbedDataset(
            csv_file=EMBED_META/ Path("embed_train_manual.csv") if self.args.use_manual else EMBED_META/ Path("embed_train.csv"),
            target_size=self.target_size,
            transform=None,
            parents=self.parents,
        )

        print("Validation df: ")
        self.dataset_val = EmbedDataset(
            csv_file=EMBED_META/ Path("embed_val_manual.csv") if self.args.use_manual else EMBED_META/ Path("embed_val.csv"),
            target_size=self.target_size,
            transform=None,
            parents=self.parents,
        )

        print(f"Test df: {EMBED_META/ Path('embed_test_manual.csv') if self.args.use_manual else EMBED_META/ Path('embed_test.csv')}")
        self.dataset_test = EmbedDataset(
            csv_file=EMBED_META/ Path("embed_test_manual.csv") if self.args.use_manual else EMBED_META/ Path("embed_test.csv"),
            target_size=self.target_size,
            transform=None,
            parents=self.parents,
        )

    @property
    def dataset_name(self):
        return "embed"

