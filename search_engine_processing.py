import reading_of_data
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import OrderedDict, defaultdict


def cosine(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    return np.dot(v1, v2) / (np.sqrt(np.sum(v1 ** 2)) * np.sqrt(np.sum(v2 ** 2)))


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

        # ordered_values_reverse = {}
        # for k, v in ordered_values.items():
        #     ordered_values_reverse[v] = ordered_values_reverse.get(v, [])
        #     ordered_values_reverse[v].append(k)

    return ordered_values
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

    # document_ids = run_simple_conjunctive_query(words, tdidf_inverted_index_items)
    # tfidf = TfidfVectorizer()
    # if document_ids is not None and len(document_ids) > 0:
    #     for document_id in document_ids:
    #         list_of_words = list(words)
    #         query_scores = list()
    #         document_scores = list()
    #         voc_dic = reading_of_data.get_vocabulary_dic()  # get a dictionary from dictionary file
    #         sentence = " ".join(list_of_words)  # joining the sentence to use it later
    #         tfidf_scores = tfidf.fit_transform([sentence])
    #         feature_names = tfidf.get_feature_names()
    #         for col in tfidf_scores.nonzero()[1]:
    #             word_item = feature_names[col]
    #             tfidf_score = tfidf_scores[0, col]
    #
    #             # get term id from voc_dic
    #             term_id = list(voc_dic.keys())[list(voc_dic.values()).index(word_item)]
    #             if str(term_id) in tdidf_inverted_index_items:
    #                 list_of_documents = tdidf_inverted_index_items[str(term_id)]
    #                 if len(list_of_documents) > 0:
    #                     for key, value in list(set(list_of_documents)):
    #                         if key == document_id:
    #
    #                             break ;
