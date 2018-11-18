import json


def get_vocabulary_dic(file_name='vocabulary.json'):  # reading of vocabulary dictionary file

    # Create an empty set for the vocabulary
    with open(file_name, 'r') as fp:
        data = json.load(fp)
    return data


def get_inverted_index_file(file_name='inverted_index.json"'):  # reading inverted_index dictionary file

    # Create an empty set for the vocabulary
    with open(file_name, 'r') as fp:
        data = json.load(fp)
    return data

