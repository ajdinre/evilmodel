from transformers import AutoImageProcessor, AutoModelForImageClassification
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import struct
import torchvision.models as models


def change_rightmost_byte(byte_object):
    return byte_object[:-1] + b"\xFF"


processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = models.resnet50()  # Initialize the ResNet50 model
model.load_state_dict(
    torch.load("./models/resnet50-19c8e357.pth")
)  # Load the weights from the specified file
model.eval()  # Set the model to evaluation mode


first_layer_printed = False
for name, module in model.named_modules():
    if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
        if isinstance(module, torch.nn.Conv2d):
            num_neurons = module.out_channels
        else:
            num_neurons = module.out_features
        print(f"Layer: {name}, Type: {type(module).__name__}, Neurons: {num_neurons}")

        if not first_layer_printed:
            print(f"\nNeurons in the first layer ({name}):")
            weights = module.weight.data.flatten().tolist()
            modified_weights = []
            for weight in weights:
                print(f"Original weight: {weight}")
                packed = struct.pack("!f", weight)
                print(f"Packed bytes: {packed.hex()}")

                modified = change_rightmost_byte(packed)
                print(f"Modified bytes: {modified.hex()}")

                unpacked = struct.unpack("!f", modified)[0]
                print(f"Unpacked modified weight: {unpacked}")
                print()
                modified_weights.append(unpacked)

            first_layer_printed = True
            with torch.no_grad():
                module.weight.data = torch.tensor(modified_weights).reshape(
                    module.weight.shape
                )


dataset = load_dataset(
    "cifar10", split="train[:1%]"
)  # using a subset for demonstration


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
    num_proc=4,
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
            "labels": torch.tensor(item["labels"]).squeeze(),
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
        outputs = model(batch["pixel_values"])  # Pass the tensor directly
        loss = loss_fn(outputs, batch["labels"])  # Adjusted to match the output format
        total_loss += loss.item()
        num_batches += 1

    progress_bar.set_postfix({"Avg Loss": f"{total_loss / num_batches:.4f}"})

average_loss = total_loss / num_batches
print(f"\nFinal average loss: {average_loss:.4f}")

# Save the modified model to a new file
torch.save(
    model.state_dict(), "./models/resnet50_modified.pth"
)  # Save the modified model weights
