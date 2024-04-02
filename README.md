# MalBERT Project

## Introduction
The MalBERT project is a machine learning-based approach for malware detection using the BERT model. This README provides instructions on how to set up and run the project.

## Prerequisites
- Python 3.7 or higher
- TensorFlow 2.0 or higher
- Other dependencies listed in requirements.txt

## Setup
1. Clone the MalBERT repository.
2. Place the training samples (APKs) inside the `data/aps/D0/goodware` and `data/aps/D0/malware` directories.
3. Run the `decompress_apks.sh` script to decompress the APKs.
4. Run the `fetch_manifests.sh` script to fetch the APK manifests.
5. Run the `preprocessing.py` script to preprocess the data.

## Training
1. Run the `train.py` script to train the MalBERT model.
2. Monitor the training progress and adjust hyperparameters as needed.

## Evaluation
1. Use the trained model to evaluate the performance on test data.
2. Analyze the results and fine-tune the model if necessary.

## Contributing
Contributions are welcome! Please follow the guidelines in the CONTRIBUTING.md file.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

## Contact
For any questions or issues, please contact the project maintainers at simon.queyrut@unine.ch.