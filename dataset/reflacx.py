import os
import torch
import albumentations

import numpy as np
import pandas as pd

from . import constants
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision.ops.boxes import box_area
from torchvision.models import VisionTransformer
from sklearn.calibration import LabelEncoder


def box_xyxy_to_cxcywh(x):
    if len(x) == 0:
        return x
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


training_clinical_mean_std = {
    "age": {"mean": 62.924050632911396, "std": 18.486667896662354},
    "temperature": {"mean": 98.08447784810126, "std": 2.7465209372955712},
    "heartrate": {"mean": 85.95379746835444, "std": 18.967507646992733},
    "resprate": {"mean": 18.15221518987342, "std": 2.6219004903965004},
    "o2sat": {"mean": 97.85411392405064, "std": 2.6025150031174946},
    "sbp": {"mean": 133.0685126582279, "std": 25.523304795054102},
    "dbp": {"mean": 74.01107594936708, "std": 16.401336318103716},
    "acuity": {"mean": 2.2610759493670884, "std": 0.7045539799670345},
    "pain": {"mean": 3.892105263157895, "std": 4.273759847021344},
}

class REFLACXDataset(Dataset):
    def __init__(
        self,
        df_path,
        mimic_eye_path,
        image_size,
        split_str,
        label_cols=constants.TOP5_LABEL_COLS,
        chexpert_label_cols=constants.CHEXPERT_LABEL_COLS,
        clinical_labels=[
            "temperature",
            "heartrate",
            "resprate",
            "o2sat",
            "sbp",
            "dbp",
        ],
        transform=None,
        cxcywh=True,
    ) -> None:
        super().__init__()
        self.df_path = df_path
        self.mimic_eye_path = mimic_eye_path
        self.image_size = image_size
        self.split_str = split_str
        self.label_cols = label_cols
        self.chexpert_label_cols = chexpert_label_cols
        self.cxcywh = cxcywh

        self.clinical_num = [
            "age",
            # "temperature",
            # "heartrate",
            # "resprate",
            # "o2sat",
            # "sbp",
            # "dbp",
            # "pain",
            # "acuity",
        ]
        self.clinical_cat = ["gender"]
        self.clinical_labels = clinical_labels
        self.normalise_clinical_num = True

        self.df = pd.read_csv(self.df_path)
        self.__preprocess_clinical_df()
        self.__preprocess_label()
        self.__init_transform(transform)

    def __init_transform(self, transform):
        if not transform is None:
            self.transform = transform
        else:
            self.transform = albumentations.Compose(
                [
                    albumentations.Resize(self.image_size, self.image_size),
                    albumentations.HorizontalFlip(p=0.5),
                ],
                bbox_params=albumentations.BboxParams(
                    format="pascal_voc", label_fields=["label"]
                ),
            )

    def __preprocess_clinical_df(
        self,
    ):
        self.encoders_map = {}

        # encode the categorical cols.
        for col in self.clinical_cat:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col])
            self.encoders_map[col] = le

        if self.normalise_clinical_num:
            self.clinical_std_mean = {}
            for col in [
                "age",
                "temperature",
                "heartrate",
                "resprate",
                "o2sat",
                "sbp",
                "dbp",
                "pain",
                "acuity",
            ]:
                # calculate mean and std
                mean = training_clinical_mean_std[col]["mean"]
                std = training_clinical_mean_std[col]["std"]
                self.df[col] = (self.df[col] - mean) / std

    def __get_paths(self, data):
        reflacx_id = data["id"]
        patient_id = data["subject_id"]
        study_id = data["study_id"]
        dicom_id = data["dicom_id"]
        image_path = os.path.join(
            self.mimic_eye_path,
            f"patient_{patient_id}",
            "CXR-JPG",
            f"s{study_id}",
            f"{dicom_id}.jpg",
        )
        bbox_path = os.path.join(
            self.mimic_eye_path,
            f"patient_{patient_id}",
            "REFLACX",
            "main_data",
            reflacx_id,
            "anomaly_location_ellipses.csv",
        )

        report_path = os.path.join(
            self.mimic_eye_path,
            f"patient_{patient_id}",
            "CXR-DICOM",
            f"s{study_id}.txt",
        )
        return image_path, bbox_path, report_path

    def __get_bb_df(self, bbox_path, img_height, img_width):
        bb_list = []
        bbox_df = pd.read_csv(bbox_path)
        for i, bb in bbox_df.iterrows():
            for l in [
                col for col in bb.keys() if not col in constants.DEFAULT_BOX_FIX_COLS
            ]:
                if bb[l] == True:
                    label = constants.DEFAULT_REPETITIVE_LABEL_REVERSED_MAP[l]
                    if label in self.label_cols:
                        xmax = np.clip(bb["xmax"], 0, img_width)
                        xmin = np.clip(bb["xmin"], 0, img_width)
                        ymax = np.clip(bb["ymax"], 0, img_height)
                        ymin = np.clip(bb["ymin"], 0, img_height)

                        # width = xmax-xmin
                        # height = ymax-ymin
                        # assert width >= 0, f"Width of BB should > 0, but got [{width}]"
                        # assert height >= 0, f"Height of BB should > 0, but got [{height}]"
                        # if width * height > 0:

                        bb_list.append(
                            {
                                "x_min": xmin,
                                "y_min": ymin,
                                "x_max": xmax,
                                "y_max": ymax,
                                "label": self.lesion_to_idx(label),
                            }
                        )

        return pd.DataFrame(
            bb_list, columns=["x_min", "y_min", "x_max", "y_max", "label"]
        )

    def __get_bb_label(self, bbox_path, img_height, img_width):
        bb_df = self.__get_bb_df(bbox_path, img_height, img_width)
        bbox = torch.tensor(
            np.array(bb_df[["x_min", "y_min", "x_max", "y_max"]], dtype=float)
        )
        label = torch.tensor(np.array(bb_df["label"]).astype(int), dtype=torch.int64)

        return {
            "bbox": bbox,
            "label": label,
        }

    def __preprocess_label(
        self,
    ):
        self.df[constants.ALL_LABEL_COLS] = self.df[constants.ALL_LABEL_COLS].gt(0)
        # self.df["gender"] = self.df["gender"] == "F"

    def __prepare_chexpert_label(self, data):
        return torch.tensor(data[self.chexpert_label_cols]) == 1

    def __prepare_clinical(self, data):
        clinical_num = None
        if not self.clinical_num is None and len(self.clinical_num) > 0:
            clinical_num = torch.tensor(
                np.array(data[self.clinical_num], dtype=float)
            ).float()

        clinical_cat = None
        if not self.clinical_cat is None and len(self.clinical_cat) > 0:
            clinical_cat = {
                c: torch.tensor(np.array(data[c], dtype=int)) for c in self.clinical_cat
            }

        return {"cat": clinical_cat, "num": clinical_num}

    def __prepare_clinical_label(self, data):
        return torch.tensor(data[self.clinical_labels], dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        # determine paths
        image_path, bbox_path, report_path = self.__get_paths(data)

        xray = read_image(image_path)  # grey, c = 1, (C, H, W)
        img_height, img_width = xray.shape[1], xray.shape[2]

        bb_label = self.__get_bb_label(bbox_path, img_height, img_width)
        # num_objs = len(bb_label["bbox"])

        self.xray = xray

        transformed = self.transform(
            image=xray.repeat(3, 1, 1).permute(1, 2, 0).cpu().numpy(),
            bboxes=bb_label["bbox"],
            label=bb_label["label"],
        )
        xray = torch.tensor(transformed["image"]).permute(2, 0, 1) / 255
        boxes = torch.tensor(transformed["bboxes"]).float()  # x1,y1,x2,y2

        # assuming that we are having square images.
        if self.cxcywh:
            boxes = box_xyxy_to_cxcywh(boxes) / self.image_size

        bb_label = {
            # "image_id": idx,
            "boxes": boxes if len(boxes) > 0 else torch.zeros((0, 4)).float(),
            "labels": torch.tensor(transformed["label"], dtype=torch.int64),
            # "area": box_area(boxes) if len(boxes) > 0 else torch.zeros((0, 4)).float(),
            # "iscrowd": torch.zeros((num_objs,), dtype=torch.int64),
            # "orig_size": torch.tensor(
            # [self.image_size, self.image_size], dtype=torch.int64
            # ),
        }

        chexpert_label = self.__prepare_chexpert_label(data)

        # clinical data
        clinical_data = self.__prepare_clinical(data)

        # radiology report
        with open(report_path) as f:
            report = f.read()
        report = report.strip().replace("FINAL REPORT\n", "").replace("\n", "").strip()

        clinical_label = self.__prepare_clinical_label(data)

        return {
            "xray": xray,
            "bb_label": bb_label,
            "clinical_data": clinical_data,
            "chexpert_label": chexpert_label,
            "report": report,
        }, clinical_label

    def lesion_to_idx(self, disease: str) -> int:
        if not disease in self.label_cols:
            raise Exception("This disease is not the label.")

        if disease == "background":
            return 0

        return self.label_cols.index(disease) + 1

    def idx_to_lesion(self, idx: int) -> str:
        if idx == 0:
            return "background"

        if idx > len(self.label_cols):
            return f"exceed label range :{idx}"

        return self.label_cols[idx - 1]
