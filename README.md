# MalBERT Project

## Introduction
Implements preprocessing and training of [MalBERT](https://ieeexplore.ieee.org/document/9659287) but using [Longformers](https://huggingface.co/docs/transformers/model_doc/longformer) instead. Uses additional info from [MalBERTv2](https://www.mdpi.com/2504-2289/7/2/60).

## Prerequisites
- Python 3.7 or higher
- Torch 2.0.1
- Transformers 4.37.0

## Setup
1. Clone the MalBERT repository.
2. Place the training samples (APKs) inside `data/aps/D0/goodware` and `data/aps/D0/malware` directories (create them if needed).
3. Run the following command to decompress the APKs:
```bash
sh decompress_apks.sh
```
4. Run the `fetch_manifests.sh` script to place the manifests in the correct folder.
```bash
sh fetch_manifests.sh
```
5. Run the `preprocessing.py` script to preprocess the data (currently only handles training data).
```python
python preprocessing.py
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