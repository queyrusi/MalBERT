"""
This module preprocesses XML manifests to text files.

The module includes functions to extract relevant information from XML files, read and process XML manifests
from a directory, and save the extracted information to text files.

Usage: python preprocessing.py <manifests_dir> <output_dir>
"""
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
import re
from xml.dom import minidom
import argparse
from tqdm import tqdm
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


def extract_manifest_info(xml_file, dictionary_path):
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

    dictionary_file = dictionary_path
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


def main(manifests_dir, output_dir, dictionary_path):
    # Read and process goodware XML manifests
    for file in tqdm(os.listdir(manifests_dir), desc="tokenizing manifests"):
        try:
            if file.endswith('.xml'):
                filename = os.path.splitext(file)[0]
                output_file = os.path.join(output_dir, filename)
                if not os.path.exists(output_file):
                    manifest_info = extract_manifest_info(os.path.join(manifests_dir, file), dictionary_path)
                    manifest_string = ' '.join(manifest_info)
                    with open(output_file, 'w') as f:
                        f.write(manifest_string)
        except Exception as e:
            print(f"Error processing file {file}: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process XML manifests.')
    parser.add_argument('manifests_dir', type=str, help='Path to the directory containing XML manifests')
    parser.add_argument('output_dir', type=str, help='Path to the output directory')
    parser.add_argument('dictionary_path', type=str, help='Path to the dictionary file')
    args = parser.parse_args()

    manifests_dir = args.manifests_dir
    output_dir = args.output_dir
    dictionary_path = args.dictionary_path

    main(manifests_dir, output_dir, dictionary_path)
