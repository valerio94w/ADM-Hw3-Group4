import reading_of_data
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import OrderedDict, defaultdict
import heapq


def cosine(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    prodouct_of_both = np.dot(v1, v2)
    square_root_of_v1 = np.sqrt(np.sum(v1 ** 2))
    square_root_of_v2 = np.sqrt(np.sum(v2 ** 2))
    prodcuct_of_square_root_of_both = (square_root_of_v1 * square_root_of_v2)
    cosine_value = (prodouct_of_both / prodcuct_of_square_root_of_both)
    return cosine_value


def run_simple_conjunctive_query(words, inverted_index_items):
    query = list(words)

    # Each of the querys matches should contain all the words on the list query
    voc_dic = reading_of_data.get_vocabulary_dic()  # get a dictionary from dictionary file
    querys_matches = []
    for word in query:
        term_id = list(voc_dic.keys())[list(voc_dic.values()).index(word)]
        if term_id in inverted_index_items.keys():
            querys_matches.append(set(inverted_index_items[term_id]))
    # As a result we are going to have a list, with all the matches, called inter
    k = 0
    if len(querys_matches) > 0:
        inter = querys_matches[k]
        k += 1
        for i in range(1, len(querys_matches)):
            inter = inter.intersection(querys_matches[k])
            k += 1
    return inter


def run_cosine_similarity_tfidf_conjunctive_query(words, tdidf_inverted_index_items):
    query = list(words)

    # Each of the querys matches should contain all the words on the list query
    voc_dic = reading_of_data.get_vocabulary_dic()  # get a dictionary from dictionary file
    querys_matches = []
    for word in query:
        term_id = list(voc_dic.keys())[list(voc_dic.values()).index(word)]
        if term_id in tdidf_inverted_index_items.keys():
            querys_matches.append(list(tdidf_inverted_index_items[term_id]))
    # As a result we are going to have a list, with all the matches, called inter

    k = 0
    # documents_dictionary_with_tfidf = defaultdict(list)

    if len(querys_matches) > 0:
        inter = return_set_of_docs_from_tfidf_inverted_index_doc_item(querys_matches[k])
        k += 1
        for i in range(1, len(querys_matches)):
            inter = inter.intersection(return_set_of_docs_from_tfidf_inverted_index_doc_item(querys_matches[k]))
            k += 1

        documents_dictionary_with_tfidf = []

        for i in range(0, len(querys_matches)):
            common_docs = inter.intersection(return_set_of_docs_from_tfidf_inverted_index_doc_item(querys_matches[i]))
            documents_scores = get_tdidf_values_from_doc_ids(querys_matches[i], common_docs)
            documents_dictionary_with_tfidf.append(documents_scores)

        # I have 2D arry for each document in which columns contains tfidf values
        documents_dictionary_with_tfidf_matrix = np.array(documents_dictionary_with_tfidf)

        list_of_scores_for_query = []
        # I have to calculate the tdidf values for current sentence
        list_of_words = words
        voc_dic = reading_of_data.get_vocabulary_dic()  # get a dictionary from dictionary file
        voc_items = " ".join(voc_dic.values())
        sentence = " ".join(list_of_words)  # joining the sentence to use it later
        tfidf = TfidfVectorizer()
        tfidf_scores = tfidf.fit_transform([sentence])
        feature_names = tfidf.get_feature_names()
        for col in tfidf_scores.nonzero()[1]:
            word_item = feature_names[col]
            tfidf_score = tfidf_scores[0, col]
            list_of_scores_for_query.append(tfidf_score)

        # now I am going to calculate cosine similarity
        cosine_similarity_dictionary = defaultdict(str)
        for i in range(len(documents_dictionary_with_tfidf_matrix[0])):
            doc_n = documents_dictionary_with_tfidf_matrix[:, i]
            cosine_similarity_value = cosine(np.array(list_of_scores_for_query), doc_n)
            cosine_similarity_dictionary[list(common_docs)[i]] = cosine_similarity_value

        # sorting dictionary by value

        ordered_values = OrderedDict(sorted(cosine_similarity_dictionary.items(), key=lambda x: x[1]))

        priority_queue = []
        # priority queue
        for key, value in ordered_values.items():
            heapq.heappush(priority_queue,  (value * -1, key))

    return priority_queue
    # now we got the docs_ids which matched


def get_tdidf_values_from_doc_ids(items, common_docs):
    doc_ids = list()
    if len(items) > 0:
        for item in items:
            for key, value in item.items():  # iterating through dictionary it will only have one element
                for doc_id in common_docs:
                    if doc_id == key:
                        doc_ids.append(value)
                        break
    return doc_ids


def return_set_of_docs_from_tfidf_inverted_index_doc_item(items):
    doc_ids = set()
    if len(items) > 0:
        for item in items:
            for key in item.keys():  # iterating through dictionary
                doc_ids.add(key)
    return doc_ids


