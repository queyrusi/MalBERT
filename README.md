# MalBERT Project

## Introduction
Implements preprocessing and training of [MalBERT](https://ieeexplore.ieee.org/document/9659287) but using [Longformers](https://huggingface.co/docs/transformers/model_doc/longformer) instead. Uses additional info from [MalBERTv2](https://www.mdpi.com/2504-2289/7/2/60).

## Prerequisites
- Python 3.7 or higher
- Torch 2.0.1
- Transformers 4.37.0

## Setup
1. Clone the MalBERT repository.
2. Run the `get_manifests.sh` script to extract manifests.
```bash
sh get_manifests.sh "path/to/apk/folder" "path/to/output/folder"
```
Extract them to goodware and malware subfolders.

3. Run the `preprocessing.py` script to preprocess the manifests to txt.
```python
python preprocessing.py "path/to/manifest/folder" "path/to/output/folder"
````

## Training
Run the `train.py` script to train the MalBERT model (uses Longformer).
```python
python train.py
```

## Evaluation
Test part is commented inside the `train.py` file but it will have its own module.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact
For any questions or issues, please contact the project maintainers at simon.queyrut@unine.ch.