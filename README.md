# Homework 3 - ADM - Group4

This repository contains the Homework 3 of Algorithmic Methods of Data Mining Academic year 2018–2019 from group 4 formed by Valerio Antonini, Hafiz Muhammad Hassan and Francisca Alliende.

Incoming, a summary of the files of this repository:

### 1. **[Homework_3.ipynb](http://nbviewer.jupyter.org/github/valerio94w/ADM-Hw3-Group4/blob/master/Homework_3.ipynb)** 
A Jupyter Notebook with the results of the analisys of Airbnb property listings and the search engines requested.

### 2. **Functions files**:
- **[cleaning_of_data.py](https://github.com/valerio94w/ADM-Hw3-Group4/blob/master/cleaning_of_data.py):** this file contains the necessary functions to clean the data, such as removing null values and irrelevant    characters, among others.
- **writing_of_data.py:** this file contains the functions used to create data, such as the .tsv files of the Step 2, creating, the inverted indexes, the vocabulary file, and the output results.  
- **reading_of_data.py:** this file contains the necessary functions to reading the files created in previous steps i.e. inverted_index.json, inverted_index_tfidf.json and vocabulary.json.
- **search_engine.py:** this file contains all the necessary functions to execute the searching engine, from the query to the results.
- **search_engine_processing.py:** this file contains all the functiosn relatives to the querys, such that the cosine similarity, the conjunctive query, tdidf similarity, among others.

### 3. **Databases, inverted indexes and vocabulary:**
- **Airbnb Texas Rentals.csv:** a csv file that corresponds to the database used to make the analyzes and the search engines
- **inverted_index.json:** a json file that contains the inverted index created during the process of making the conjunctive query of the Section 3.1.
- **inverted_index_tfidf.json:** a json file that contains the inverted index created during the process of making the conjunctive query & rnking score of the Section 3.2.
- **vocabulary.json:** a json file that contains the vocabulary, that maps each word in the database to an ID.

