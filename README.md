# EvilModel

This is an implementation of 'EvilModel: Hiding Malware Inside of Neural Network Models' paper. It explores embedding arbitrary data such as malware inside NN. Since the data is split around the file as weights, it's hard to detect.

## How It Works

The system consists of two main scripts:

1. `embed.py`: Embeds a file into a PyTorch model's weights.
2. `extract.py`: Extracts the embedded file from a modified PyTorch model.

The embedding process works by replacing the least significant byte of each float in the model's weights with a byte from the file to be embedded. An escape sequence is added at the end of the file to mark its end.

The extraction process reverses this, collecting the least significant bytes from the model's weights until it encounters the escape sequence.

## Usage

### Embedding a file

```bash
$ python embed.py --model resnet50.pth --file not_malware --output evil_resnet.pth
Loading model from resnet50.pth
Embedding file not_malware into model weights
File size: 3619125 bytes
Total model parameters: 25610152
Using escape sequence: deadbeef
Successfully embedded 3619129 bytes into the model
Embedding complete: True
Saving modified model to evil_resnet.pth
Operation completed successfully
```

### Extracting a file

```bash
$ python extract.py --model evil_resnet.pth --output extracted_not_malware        
Loading model from evil_resnet.pth
Extracting file from model weights
Extracted 3619125 bytes to extracted_not_malware
Operation completed successfully
```

## Results for ResNet
TODO: add change in eval score


## Files are the same

```
(venv) dino@tiny evilmodel % shasum -a 256 *not_malware
1c67c7ceb85301c8578bd87e9177f0deee8a306692096ab31d42d2588d9293c8  extracted_not_malware
1c67c7ceb85301c8578bd87e9177f0deee8a306692096ab31d42d2588d9293c8  not_malware
```