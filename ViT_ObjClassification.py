# Description :  Object Classification using Visual Transformer Architecture
# Date : 9/21/2024
# Author : Dude
# URLs :
#           https://huggingface.co/blog/fine-tune-vit
#   Dataset : Oxford Cats and Dogs dataset
#          https://www.kaggle.com/datasets/karakaggle/kaggle-cat-vs-dog-dataset
# Problems / Solutions :
#   P1 : No object 'label' in transform
#   S1 :  inputs["labels"] = sample_batch["label"] <== The "imagefolder" dataset type creates a 'label' feature whereas preprocessor wanted 'labels'
# Revisions :
#
from CnD_Dataset import CnDs
import torch
import numpy as np
from datasets import load_metric
from transformers import ViTForImageClassification
from transformers import TrainingArguments, Trainer
from transformers import pipeline
from PIL import Image

DATA_PATH = "C:\Kuljeet\WorkSpace\PyTorch\ViT_ObjClassification\data\CnDs"
TEST_IMAGE_1 = "C:\Kuljeet\WorkSpace\PyTorch\ViT_ObjClassification\data\evaluation\cat_or_dog_1.jpg"
TEST_IMAGE_2 = "C:\Kuljeet\WorkSpace\PyTorch\ViT_ObjClassification\data\evaluation\cat_or_dog_2.jpg"
SAVED_MODEL = "vit_cnds"
# There is also google/vit-base-patch32-224-in21k and  google/vit-base-patch32-384
# Images are presented to the model as a sequence of fixed-size patches (resolution 16x16), which are linearly embedded.
MODEL_PATH = "google/vit-base-patch16-224-in21k"


def collate_fn(batch):
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


def train_n_eval():
    # Load the custom dataset in hugging face format
    imgdatasets = CnDs(DATA_PATH, MODEL_PATH, desired_size=(224))
    print(imgdatasets.dataset["train"][0])
    metric = load_metric("accuracy")

    def compute_metrics(p):
        return metric.compute(
            predictions=np.argmax(p.predictions, axis=1), references=p.label_ids
        )

    labels = imgdatasets.get_class_labels()
    # Load pretrained model
    model = ViTForImageClassification.from_pretrained(
        MODEL_PATH,
        num_labels=len(labels),
        id2label={str(i): c for i, c in enumerate(labels)},
        label2id={c: str(i) for i, c in enumerate(labels)},
    )
    # Preprare training parameters
    training_args = TrainingArguments(
        output_dir="./output",
        per_device_train_batch_size=16,  # aligned to patch count
        eval_strategy="steps",
        num_train_epochs=5,
        fp16=True,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=2e-5,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to="tensorboard",
        load_best_model_at_end=True,
    )
    # Configure trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
        train_dataset=imgdatasets.prepared_ds["train"],  # use pixel_values as input
        eval_dataset=imgdatasets.prepared_ds["test"],  # use pixel_values as input
        tokenizer=imgdatasets.processor,
    )
    # Train model
    train_results = trainer.train()
    # Save model
    trainer.save_model(SAVED_MODEL)
    trainer.log_metrics("train", train_results.metrics)
    trainer.save_metrics("train", train_results.metrics)
    trainer.save_state()
    # Evaluate model
    metrics = trainer.evaluate(imgdatasets.prepared_ds["test"])
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


# Predict one image
def predict_one_image(image_path):
    image = Image.open(image_path)

    # Load Saved Model
    from transformers import ViTImageProcessor, ViTModel

    processor = ViTImageProcessor.from_pretrained(SAVED_MODEL, local_files_only=True)
    # For feature extraction
    #    model = ViTModel.from_pretrained(SAVED_MODEL, local_files_only=True)
    # For classification
    model = ViTForImageClassification.from_pretrained(
        SAVED_MODEL, local_files_only=True
    )
    inputs = processor(images=image, return_tensors="pt")

    outputs = model(**inputs)
    # last_hidden_states = outputs.last_hidden_state
    # print(last_hidden_states) # For feature extraction task
    logits = outputs.logits  # For classification  task
    predicted_class_idx = logits.argmax(-1).item()
    print("Predicted class:", model.config.id2label[predicted_class_idx])

    # For feature extraction task
    # pipe = pipeline(
    #     task="image-feature-extraction",
    #     model=model,
    #     framework="pt",
    #     pool=True,
    # )

    # For classification task
    pipe = pipeline(
        task="image-classification",
        model=model,
        image_processor=processor,
        framework="pt",
    )
    print(pipe(image_path))


if __name__ == "__main__":
    train_n_eval()
    predict_one_image(TEST_IMAGE_2)
