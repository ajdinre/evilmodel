# recon.py

import torch
import pickle
import os
import argparse


def is_torch_model(filepath):
    try:
        model = torch.load(
            filepath, map_location=torch.device("cpu"), weights_only=True
        )
        if isinstance(model, torch.nn.Module) or isinstance(model, dict):
            return True, model
        return False, None
    except Exception:
        return False, None


def print_model_info(model, model_type):
    print(f"Model Type: {model_type}")
    if model_type == "PyTorch":
        if isinstance(model, torch.nn.Module):
            print(f"Model Class: {model.__class__.__name__}")
            print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
        elif isinstance(model, dict):
            print(f"Model contains {len(model.keys())} items.")


def main(filepath):
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist.")
        return

    is_torch, model = is_torch_model(filepath)
    if is_torch:
        print_model_info(model, "PyTorch")
        return

    print("Model type not recognized or unsupported.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inspect a model file and detect if it's PyTorch, TensorFlow, or JAX."
    )
    parser.add_argument(
        "filepath", type=str, help="The path to the model file to be inspected."
    )

    args = parser.parse_args()
    main(args.filepath)
