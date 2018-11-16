import pandas as pd
import csv
import nltk
import os
from pandas import DataFrame
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer

import cleaning_of_data

CSV_FILE_NAME = "Airbnb_Texas_Rentals.csv"  # constant csv file name
FOLDER_NAME_FOR_TSV_File = "doc_files"
VOC_FILE_NAME = "vocabulary.csv"

# cleaning of data
df = cleaning_of_data.open_csv_file_and_remove_extra_values(CSV_FILE_NAME)
cleaning_of_data.create_tsv_files(df, FOLDER_NAME_FOR_TSV_File)

# applying NLTK techniques

cleaning_of_data.create_vocabulary_from_data_frame(df, FOLDER_NAME_FOR_TSV_File, VOC_FILE_NAME)


