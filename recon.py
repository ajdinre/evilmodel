# recon.py

import torch
import tensorflow as tf
import jax
import pickle
import os
import argparse


def is_torch_model(filepath):
    try:
        model = torch.load(filepath, map_location=torch.device('cpu'))
        if isinstance(model, torch.nn.Module) or isinstance(model, dict):
            return True, model
        return False, None
    except Exception:
        return False, None


def is_tensorflow_model(filepath):
    try:
        model = tf.keras.models.load_model(filepath)
        return True, model
    except Exception:
        return False, None


def is_jax_model(filepath):
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        if isinstance(model, dict) and 'params' in model:
            return True, model
        return False, None
    except Exception:
        return False, None


def print_model_info(model, model_type):
    print(f"Model Type: {model_type}")
    if model_type == 'PyTorch':
        if isinstance(model, torch.nn.Module):
            print(f"Model Class: {model.__class__.__name__}")
            print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
        elif isinstance(model, dict):
            print(f"Model contains {len(model.keys())} items.")
    elif model_type == 'TensorFlow':
        model.summary()
    elif model_type == 'JAX':
        print(f"Model Parameters: {model.get('params')}")
        print(f"Number of parameters: {sum(x.size for x in jax.tree_util.tree_leaves(model['params']))}")


def main(filepath):
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist.")
        return

    is_torch, model = is_torch_model(filepath)
    if is_torch:
        print_model_info(model, 'PyTorch')
        return

    is_tf, model = is_tensorflow_model(filepath)
    if is_tf:
        print_model_info(model, 'TensorFlow')
        return

    is_jax, model = is_jax_model(filepath)
    if is_jax:
        print_model_info(model, 'JAX')
        return

    print("Model type not recognized or unsupported.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a model file and detect if it's PyTorch, TensorFlow, or JAX.")
    parser.add_argument("filepath", type=str, help="The path to the model file to be inspected.")

    args = parser.parse_args()
    main(args.filepath)

