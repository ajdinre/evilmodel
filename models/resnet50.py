from transformers import AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm


processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = AutoModelForImageClassification.from_pretrained("microsoft/resnet-50", cache_dir="models/resnet50", force_download=True)

dataset = load_dataset("cifar10", split="train[:1%]")  # using a subset for demonstration

def preprocess_function(examples):
    images = [np.array(img.convert("RGB")) for img in examples["img"]]
    inputs = processor(images=images, return_tensors="pt")
    inputs["labels"] = torch.tensor(examples["label"])
    return inputs

processed_dataset = dataset.map(
    preprocess_function,
    batched=True,
    remove_columns=dataset.column_names,
    desc="Preprocessing",
    num_proc=4 
)


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        return {
            "pixel_values": torch.tensor(item["pixel_values"]).squeeze(),
            "labels": torch.tensor(item["labels"]).squeeze()
        }

tensor_dataset = TensorDataset(processed_dataset)

dataloader = DataLoader(tensor_dataset, batch_size=8)

model.eval()
loss_fn = torch.nn.CrossEntropyLoss()

total_loss = 0
num_batches = 0

progress_bar = tqdm(dataloader, desc="Evaluating", unit="batch")

for batch in progress_bar:
    with torch.no_grad():
        outputs = model(pixel_values=batch["pixel_values"])
        loss = loss_fn(outputs.logits, batch["labels"])
        total_loss += loss.item()
        num_batches += 1
    
    progress_bar.set_postfix({"Avg Loss": f"{total_loss / num_batches:.4f}"})

average_loss = total_loss / num_batches
print(f"\nFinal average loss: {average_loss:.4f}")