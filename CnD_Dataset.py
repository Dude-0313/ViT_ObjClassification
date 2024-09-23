# Description : Cats and Dogs Dataset in Huggingface format
# Date : 9/21/2024 (21)
# Author : Dude
# URLs :
#
# Problems / Solutions :
#
# Revisions :
#

from datasets import load_dataset, load_metric, DatasetDict
from transformers import ViTImageProcessor
import torchvision.transforms as transforms


class CnDs:
    def __init__(self, data_path, model_name, desired_size=(224)):
        self.folder = data_path
        self.dataset = load_dataset(
            "imagefolder", data_dir=self.folder
        )  # "imagefolder" automatically creates label features based on folder hierarchy
        self.model_name = model_name
        self.desired_size = desired_size
        print(self.dataset["train"].features)
        print(self.dataset["train"].features["label"].names)
        print(self.dataset.shape)
        self.processor = ViTImageProcessor.from_pretrained(self.model_name)
        self.prepared_ds = self.dataset.with_transform(self.transform)

    def __len__(self):
        return len(self.datset)

    def __getitem__(self, index):
        image, label = self.datset[index]
        return image, label

    def get_class_labels(self):
        return self.dataset["train"].features["label"].names

    def transform(self, sample_batch):
        # Resize Image used desired_size = (W,H) or (S - to keep aspect ratio, smaller side will be resized to S the other side will be kept align to aspect ratio)
        desired_size = self.desired_size  # (244) or (224,244)
        # resize images for the sample batch
        resized_images = [
            transforms.Resize(desired_size)(x.convert("RGB"))
            for x in sample_batch["image"]
        ]
        # Convert resized images to pixel values
        inputs = self.processor(resized_images, return_tensors="pt")
        inputs["labels"] = sample_batch["label"]
        return inputs
