"""
This module contains functions for preprocessing XML manifest files.

The module includes functions to extract relevant information from XML files, read and process XML manifests,
create a DataFrame with the extracted data, split the data into training and testing sets, tokenize the text data using BERT tokenizer,
and preprocess the text data by tokenizing it using BERT tokenizer.

The preprocessed data is then saved as CSV files for training.
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
import re
from xml.dom import minidom
import xml.etree.ElementTree as ET


def maxmunch(string, dictionary):
    """
    Applies the maxmatch algorithm to a string using a dictionary of words.

    Args:
        string (str or list): The input string or list of strings to be tokenized.
        dictionary (list): A list of words in the dictionary.

    Returns:
        tokens (list): The tokenized string.
    """
    tokens = []
    if isinstance(string, str):
        string = [string]
    for word in string:
        while word:
            found = False
            for i in range(len(word), 0, -1):
                if word[:i].lower() in [word.lower() for word in dictionary]:
                    tokens.append(word[:i])
                    word = word[i:]
                    found = True
                    break
            if not found:
                tokens.append(word[0])
                word = word[1:]
    return tokens


def split_underscore(input_string):
    """
    Splits a string or a list of strings into separate words following underscore convention.

    Args:
        input_string (str or list): The input string or list of strings.

    Returns:
        words (list): The list of separate words.
    """
    words = []
    if isinstance(input_string, str):
        input_string = [input_string]
    for string in input_string:
        words.extend(string.split('_'))
    return [word for word in words]


def split_camel_case(input_string):
    """
    Splits a string or a list of strings into separate words following camel case convention.

    Args:
        input_string (str or list): The input string or list of strings.

    Returns:
        words (list): The list of separate words.
    """
    words = []
    if isinstance(input_string, str):
        input_string = [input_string]
    for string in input_string:
        word = re.sub('([A-Z][a-z]+)', r' \1', string)
        words.extend(word.split())
    return [word.lower() for word in words]


def extract_manifest_info(xml_file):
    """
    Extracts relevant information from an XML file.

    Args:
        xml_file (str): Path to the XML file.

    Returns:
        extracted_info: The extracted information from the XML file.
    """
    # Parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    extracted_info = []

    # --- Get dictionary of words

    dictionary_file = 'data/dictionaries/words.txt'
    with open(dictionary_file, 'r') as f:
        dictionary = [line.strip() for line in f]

    # --- Cleaning rules

    # Extract text and attributes
    for element in root.iter():

        # Extract text
        if element.text:
            if element.text.strip():
                lexon = element.text.strip()
                if '.' in lexon:
                    attr_values = attr_value.split('.')
                    extracted_info.extend(attr_values)
                extracted_info.append(element.text.strip())
        
        # Extract attributes
        if element.attrib:
            for attr_name, attr_value in element.attrib.items():

                if attr_name.isdigit():
                    continue

                # Replace all strange characters with _
                special_characters = ['@', ':', '/', '.', '|', '{', '}']
                for special_character in special_characters:
                    attr_name = attr_name.replace(special_character, '_')

                lexon = attr_name
                lexon = split_underscore(lexon)
                lexon = split_camel_case(lexon)
                lexon = maxmunch(lexon, dictionary)
                extracted_info.extend(lexon)

                if attr_value.isdigit():
                    continue

                # Replace all strange characters with _
                special_characters = ['@', ':', '/', '.', '|', '{', '}']
                for special_character in special_characters:
                    attr_value = attr_value.replace(special_character, '_')

                lexon = attr_value
                lexon = split_underscore(lexon)
                lexon = split_camel_case(lexon)
                lexon = maxmunch(lexon, dictionary)
                lexon = [word for word in lexon if len(word) > 1]
                extracted_info.extend(lexon)

    return extracted_info


# Define the paths to the directories containing XML manifests
goodware_dir = 'data/manifests/D0/goodware'
malware_dir = 'data/manifests/D0/malware'

# Read and process goodware XML manifests
goodware_data = []
for file in os.listdir(goodware_dir):
    if file.endswith('.xml'):
        filename = os.path.splitext(file)[0]
        output_file = f'data/train/D0/goodware/{filename}'
        if not os.path.exists(output_file):
            manifest_info = extract_manifest_info(os.path.join(goodware_dir, file))
            manifest_string = ' '.join(manifest_info)
            with open(output_file, 'w') as f:
                f.write(manifest_string)


# Read and process malware XML manifests
malware_data = []
for file in os.listdir(malware_dir):
    if file.endswith('.xml'):
        filename = os.path.splitext(file)[0]
        output_file = f'data/train/D0/malware/{filename}'
        if not os.path.exists(output_file):
            manifest_info = extract_manifest_info(os.path.join(malware_dir, file))
            manifest_string = ' '.join(manifest_info)
            with open(output_file, 'w') as f:
                f.write(manifest_string)
