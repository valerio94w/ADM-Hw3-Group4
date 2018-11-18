import reading_of_data
from sklearn.feature_extraction.text import TfidfVectorizer


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


def run_cosine_similarity_tfidf_conjunctive_query(words, inverted_index_items):
    document_ids = run_simple_conjunctive_query(words,inverted_index_items)
    tfidf = TfidfVectorizer()
    if document_ids is not None and len(document_ids) > 0:
        for document_id in document_ids:
            list_of_words = list(words)
            query_scores = list()
            document_scores = list()
            voc_dic = reading_of_data.get_vocabulary_dic()  # get a dictionary from dictionary file
            sentence = " ".join(list_of_words)  # joining the sentence to use it later
            tfidf_scores = tfidf.fit_transform([sentence])
            feature_names = tfidf.get_feature_names()
            for col in tfidf_scores.nonzero()[1]:
                word_item = feature_names[col]
                tfidf_score = tfidf_scores[0, col]

                # get term id from voc_dic
                term_id = list(voc_dic.keys())[list(voc_dic.values()).index(word_item)]
                if str(term_id) in inverted_index_items:
                    list_of_documents = inverted_index_items[str(term_id)]
                    if len(list_of_documents) > 0:
                        for key, value in list(set(list_of_documents)):
                            if key == document_id:

                                break ;


