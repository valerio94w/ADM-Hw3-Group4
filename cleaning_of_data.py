import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer


def open_csv_file_and_remove_extra_values(filename): # open csv file and clean the data

    # Import the data
    df = pd.read_csv(filename)
    df.isnull().sum()
    # Drop the rows with null numbers
    df = df[pd.notnull(df["description"])]
    df = df[pd.notnull(df["title"])]
    return df


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
    # Removing stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    # No integers
    words = [x for x in words if not (x.isdigit() or x[0].isdigit())]

    return words



