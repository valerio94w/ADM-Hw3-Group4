import pandas as pd
import pathlib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer

# open csv file and clean the data


def open_csv_file_and_remove_extra_values(filename):
    # Import the data
    df = pd.read_csv(filename)
    df.isnull().sum()
    # Drop the rows with null numbers
    df = df[pd.notnull(df["description"])]
    df = df[pd.notnull(df["title"])]
    return df


def create_tsv_files(df, folder_name):
    # Create a tsv file for each row
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)  # create a doc folder first
    for i in range(100):
        pd.DataFrame(df.iloc[i]).transpose().to_csv(folder_name + '/doc_%s.tsv' % i, sep='\t')


def create_vocabulary_from_data_frame(df, folder_name, voc_file_name):
    # Create an empty set for the vocabulary
    voc_set = set()

    # For every file...
    for i in range(100):
        doc = pd.read_csv(folder_name + '/doc_%s.tsv' % i, sep='\t')

        # Concatenate the description and title in a string
        words = doc["description"][0] + doc["title"][0]

        words = remove_extras_from_query(words)

        # Storage the words in vocabulary set
        voc_set.update(words)

    # saving voc_file
    df = pd.DataFrame(list(voc_set), columns=["title"])
    df.to_csv(voc_file_name, index=False)

    return voc_set

#
# def create__from_data_frame(df, folder_name):
#     # Create an empty set for the vocabulary
#     voc_set = set()
#
#     # For every file...
#     for i in range(len(df)):
#         doc = pd.read_csv(folder_name + '//doc_%s.tsv' % i, sep='\t')
#
#         # Concatenate the description and title in a string
#         words = doc["description"][0] + doc["title"][0]
#
#         words = remove_extras_from_query(words)
#
#         # Storage the words in vocabulary set
#         voc_set.update(words)
#
#     return voc_set


def remove_extras_from_query(query):  # use all the techniques to remove unwanted items from words

    tokenizer = RegexpTokenizer(r'\w+')

    ps = PorterStemmer()
    # Remove "\\n" and replace with a space
    words = query.replace("\\n", " ")
    # Convert everything to lowercase
    words = words.lower()
    # Removing punctuation
    words = tokenizer.tokenize(words)
    # Stemming
    words = [ps.stem(word) for word in words]
    # Keep only unique words
    words = set(words)
    # Removing stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    # No integers
    words = [x for x in words if not (x.isdigit() or x[0].isdigit())]
    # Keep only unique words
    words = set(words)

    return words



