import torch
import struct
import argparse
from pathlib import Path

ESCAPE_SEQUENCE = b'\xde\xad\xbe\xef'

def embed_byte_in_float(original_float, byte_to_embed):
    packed_float = struct.pack('!f', original_float)
    return struct.unpack('!f', packed_float[:-1] + byte_to_embed)[0]

def embed_file_in_model_weights(model_state_dict, file_to_embed_path, escape_sequence):
    with open(file_to_embed_path, 'rb') as file:
        file_content = file.read()
    
    file_size = len(file_content)
    total_model_params = sum(param.numel() for param in model_state_dict.values())
    
    print(f"File size: {file_size} bytes")
    print(f"Total model parameters: {total_model_params}")
    print(f"Using escape sequence: {escape_sequence.hex()}")
    
    if file_size + len(escape_sequence) > total_model_params:
        raise ValueError(f"File size ({file_size} bytes) exceeds available parameter space ({total_model_params} floats)")

    file_content_with_escape = file_content + escape_sequence
    embedded_byte_count = 0

    with torch.no_grad():
        for param_name, param_tensor in model_state_dict.items():
            flattened_param = param_tensor.view(-1)
            for i in range(flattened_param.size(0)):
                if embedded_byte_count < len(file_content_with_escape):
                    flattened_param[i] = embed_byte_in_float(flattened_param[i].item(), bytes([file_content_with_escape[embedded_byte_count]]))
                    embedded_byte_count += 1
                else:
                    break
            
            if embedded_byte_count >= len(file_content_with_escape):
                break

    print(f"Successfully embedded {embedded_byte_count} bytes into the model")
    print(f"Embedding complete: {embedded_byte_count == len(file_content_with_escape)}")

def main():
    parser = argparse.ArgumentParser(description="Embed a file into a PyTorch model's weights")
    parser.add_argument("--model", type=str, required=True, help="Path to the input PyTorch model")
    parser.add_argument("--file", type=str, required=True, help="Path to the file to embed")
    parser.add_argument("--output", type=str, required=True, help="Path to save the modified model")
    parser.add_argument("--escape-sequence", type=str, default="deadbeef", help="Escape sequence in hexadecimal (default: deadbeef)")
    args = parser.parse_args()

    escape_sequence = bytes.fromhex(args.escape_sequence)

    print(f"Loading model from {args.model}")
    model_state_dict = torch.load(args.model, weights_only=True)
    
    print(f"Embedding file {args.file} into model weights")
    embed_file_in_model_weights(model_state_dict, args.file, escape_sequence)
    
    print(f"Saving modified model to {args.output}")
    torch.save(model_state_dict, args.output)
    print("Operation completed successfully")

if __name__ == "__main__":
    main()