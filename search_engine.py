import cleaning_of_data
import writing_of_data
import reading_of_data
import search_engine_processing

CSV_FILE_NAME = "Airbnb_Texas_Rentals.csv"  # constant csv file name
FOLDER_NAME_FOR_TSV_File = "doc_files" # this will be used to create doc_files folder for tsv files.
VOC_FILE_NAME = "vocabulary.json"  # this will be vocabulary file for our search engine
INVERTED_INDEX_FILE_NAME = "inverted_index.json" # this will be inverted index file name
INVERTED_INDEX_TFIDF_FILE_NAME = "inverted_index_tfidf.json"

# cleaning of data
df = cleaning_of_data.open_csv_file_and_remove_extra_values(CSV_FILE_NAME)
# writing_of_data.create_tsv_files(df, FOLDER_NAME_FOR_TSV_File)  # creating tsv files inside doc_files folder

# applying NLTK techniques

# These methods are used to create vocabulary and dictionary files.
# writing_of_data.create_vocabulary_file(len(df), FOLDER_NAME_FOR_TSV_File, VOC_FILE_NAME)
# writing_of_data.create_inverted_index_file(len(df), FOLDER_NAME_FOR_TSV_File, INVERTED_INDEX_FILE_NAME)

# # Now we have our inverted_index_file
# inverted_index_dic = reading_of_data.get_inverted_index_file(INVERTED_INDEX_FILE_NAME)
# #
# query = input()
# words = cleaning_of_data.remove_extras_from_query(query)
#
# result_items = search_engine_processing.run_simple_conjunctive_query(words, inverted_index_dic)
# df = writing_of_data.output_results(FOLDER_NAME_FOR_TSV_File, result_items)
# df

# 3.2 creating second inverted_index_tfidf file

# writing_of_data.create_tfidf_inverted_index_file(len(df), FOLDER_NAME_FOR_TSV_File, INVERTED_INDEX_TFIDF_FILE_NAME)


# Now we have our inverted_index_tfidf file
inverted_index_dic = reading_of_data.get_tfidf_inverted_index_file(INVERTED_INDEX_TFIDF_FILE_NAME)

query = input()
words = cleaning_of_data.remove_extras_from_query(query)
result_items = search_engine_processing.run_cosine_similarity_tfidf_conjunctive_query(words, inverted_index_dic)
df = writing_of_data.output_results_cosine_similarity(FOLDER_NAME_FOR_TSV_File, result_items)
df