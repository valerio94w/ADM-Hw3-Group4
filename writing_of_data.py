import pathlib
import pandas as pd
import cleaning_of_data
import json
from collections import defaultdict
import reading_of_data
import heapq
from sklearn.feature_extraction.text import TfidfVectorizer


def create_tsv_files(df, folder_name):  # this function will be used to crete tsv files

    # Create a tsv file for each row
    pathlib.Path(folder_name).mkdir(parents=True, exist_ok=True)  # create a doc folder first
    for i in range(len(df)):
        pd.DataFrame(df.iloc[i]).transpose().to_csv(folder_name + '/doc_%s.tsv' % i, sep='\t')


def create_vocabulary_file(df_length, folder_name, voc_file_name):
    # Create an empty set for the vocabulary
    voc_set = set()

    # For every file...
    for i in range(df_length):
        doc = pd.read_csv(folder_name + '/doc_%s.tsv' % i, sep='\t')

        print("creating vocabulary ed index:" + str(i))
        # Concatenate the description and title in a string
        words = doc["description"][0] + doc["title"][0]

        words = cleaning_of_data.remove_extras_from_query(words)

        # Storage the words in vocabulary set
        voc_set.update(words)

    # As a result of this problem, we have a vocabulary set with unique words
    # an a dictionary, with key: number of the document values: a list of all the words (filtered) in the Airbnb post

    # Create a vocabulary dictionary from the set dictionary
    voc_dict = {}
    voc_list = list(voc_set)
    for i in range(len(voc_list)):
        voc_dict[i] = voc_list[i]

    # saving it to Json file
    with open(voc_file_name, 'w') as fp:
        json.dump(voc_dict, fp, sort_keys=True, indent=4)

    return voc_dict


def create_inverted_index_file(df_length, folder_name, dic_file_name):
    # And an empty dictionary for storage the words for each document
    dictionary = {}

    # For every file...
    for i in range(df_length):
        doc = pd.read_csv(folder_name + '/doc_%s.tsv' % i, sep='\t')
        print("creating inverted index:" + str(i))

        # Concatenate the description and title in a string
        words = doc["description"][0] + doc["title"][0]

        words = cleaning_of_data.remove_extras_from_query(words)

        # Storage the words in vocabulary set
        dictionary.update({i: words})

    # As a result of this problem, we have a vocabulary set with unique words
    # an a dictionary, with key: number of the document values: a list of all the words (filtered) in the Airbnb post

    # Create the index
    inverted_index = defaultdict(str)

    for key, value in dictionary.items():

        list_of_words = list(value)
        voc_dic = reading_of_data.get_vocabulary_dic()  # get a dictionary from dictionary file
        for value in list_of_words:
            # get term id from voc_dic
            term_id = list(voc_dic.keys())[list(voc_dic.values()).index(value)]
            if str(term_id) in inverted_index:

                inverted_index[str(term_id)].append(key)
            else:
                inverted_index[str(term_id)] = [key]

    # saving it to Json file
    with open(dic_file_name, 'w') as fp:
        json.dump(inverted_index, fp, sort_keys=True, indent=4)

    return inverted_index


def output_results(folder_name, inter):
    if inter == set():
        print("No results were found with those characteristics")
        return
    else:
        index = 0
        inter = list(inter)
        cols_of_interest = ["Title", "Description", "City", "Url"]
        if len(inter) > 0:
            doc_id = int(inter[0])
            df = pd.read_csv(folder_name + "/doc_%s.tsv" % doc_id, sep="\t")
            df = df.rename(index=str,
                           columns={'title': 'Title',
                                    "description": "Description", "city": "City", "url": 'Url'})
            df = df.filter(cols_of_interest, axis=1)
        index += 1
        for i in range(1, len(inter)):
            doc_id = int(inter[i])
            file = pd.read_csv(folder_name + "/doc_%s.tsv" % doc_id, sep="\t")
            file = file.rename(index=str, columns={'title': 'Title',
                                                   "description": "Description", "city": "City", "url": 'Url'})
            df = df.append(file.filter(cols_of_interest, axis=1), ignore_index=True, sort=False)

    df.reset_index(drop=True, inplace=True)
    # print(df.to_string())
    return df


# 3.2 creating cosine tfIdf score and saving it to file


def create_tfidf_inverted_index_file(df_length, folder_name, dic_file_name):
    # And an empty dictionary for storage the words for each document

    dictionary = {}
    dictionary_items = []
    # For every file...
    for i in range(df_length):
        doc = pd.read_csv(folder_name + '/doc_%s.tsv' % i, sep='\t')
        print("creating inverted index tfidf:" + str(i))

        # Concatenate the description and title in a string
        words = doc["description"][0] + doc["title"][0]

        words = cleaning_of_data.remove_extras_from_query(words)

        # Storage the words in vocabulary set
        dictionary.update({i: words})
        dictionary_items.append(" ".join(list(words)))

    # As a result of this problem, we have a vocabulary set with unique words
    # an a dictionary, with key: number of the document values: a list of all the words (filtered) in the Airbnb post

    # Create the index
    inverted_index = defaultdict(str)
    voc_long_string = reading_of_data.get_vocabulary_dic()
    voc_dic = reading_of_data.get_vocabulary_dic()
    file_items = " ".join(dictionary_items)
    tfidf = TfidfVectorizer(input=file_items, sublinear_tf=True)
    response = tfidf.fit_transform(dictionary_items)

    feature_names = tfidf.get_feature_names()

    for key, value in dictionary.items():
        print(" ---------- Document %s ------- " % key)
        feature_index = response[key, :].nonzero()[1]
        tfidf_scores = zip(feature_index, [response[key, x] for x in feature_index])
        for w, s in [(feature_names[i], s) for (i, s) in tfidf_scores]:
            term_id = list(voc_dic.keys())[list(voc_dic.values()).index(w)]
            if str(term_id) in inverted_index:
                inverted_index[str(term_id)].append({key: s})
            else:
                inverted_index[str(term_id)] = [{key: s}]
    # saving it to Json file
    with open(dic_file_name, 'w') as fp:
        json.dump(inverted_index, fp, sort_keys=True, indent=4)

    return inverted_index


def output_results_cosine_similarity(folder_name, priority_queue):
    df = pd.DataFrame()
    cols_of_interest = ["Title", "Description", "City", "Url", "Similarity"]
    if priority_queue:
        value, key = heapq.heappop(priority_queue)
        doc_id = int(key)
        df = pd.read_csv(folder_name + "/doc_%s.tsv" % doc_id, sep="\t")
        df = df.rename(index=str,
                       columns={'title': 'Title',
                                "description": "Description", "city": "City", "url": 'Url'})
        df = df.filter(cols_of_interest, axis=1)
        df["Similarity"] = str((-1) * value)

        while priority_queue:
            value, key = heapq.heappop(priority_queue)
            doc_id = int(key)
            file = pd.read_csv(folder_name + "/doc_%s.tsv" % doc_id, sep="\t")
            file = file.rename(index=str, columns={'title': 'Title',
                                                   "description": "Description", "city": "City", "url": 'Url'})
            file["Similarity"] = str((-1) * value)
            df = df.append(file.filter(cols_of_interest, axis=1), ignore_index=True, sort=False)

        if df is not None:
            df.reset_index(drop=True, inplace=True)
            # print(df.to_string())
            return df
        else:
            return
    else:
        print("No results were found with those characteristics")
        return


