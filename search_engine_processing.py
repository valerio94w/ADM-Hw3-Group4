import reading_of_data


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


