import torch
import struct
import argparse
from pathlib import Path

def extract_byte_from_float(embedded_float):
    packed_float = struct.pack('!f', embedded_float)
    return packed_float[-1]

def extract_file_from_model_weights(model_state_dict, output_path, escape_sequence):
    extracted_bytes = bytearray()
    escape_sequence_found = False

    for param_name, param_tensor in model_state_dict.items():
        flattened_param = param_tensor.view(-1)
        for value in flattened_param:
            byte = extract_byte_from_float(value.item())
            extracted_bytes.append(byte)
            
            if extracted_bytes[-len(escape_sequence):] == escape_sequence:
                escape_sequence_found = True
                break
        
        if escape_sequence_found:
            break

    if not escape_sequence_found:
        print("Warning: Escape sequence not found. File might be incomplete.")
    
    extracted_file_content = extracted_bytes[:-len(escape_sequence)]

    with open(output_path, 'wb') as file:
        file.write(extracted_file_content)

    print(f"Extracted {len(extracted_file_content)} bytes to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Extract an embedded file from a PyTorch model's weights")
    parser.add_argument("--model", type=str, required=True, help="Path to the input PyTorch model")
    parser.add_argument("--output", type=str, required=True, help="Path to save the extracted file")
    parser.add_argument("--escape-sequence", type=str, default="deadbeef", help="Escape sequence in hexadecimal (default: deadbeef)")
    args = parser.parse_args()

    escape_sequence = bytes.fromhex(args.escape_sequence)

    print(f"Loading model from {args.model}")
    model_state_dict = torch.load(args.model, weights_only=True)
    
    print(f"Extracting file from model weights")
    extract_file_from_model_weights(model_state_dict, args.output, escape_sequence)
    
    print("Operation completed successfully")

if __name__ == "__main__":
    main()
