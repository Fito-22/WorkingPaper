import pandas as pd
import numpy as np
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import gensim.downloader as gen_api
from tensorflow.keras.preprocessing.sequence import pad_sequences

def preprocessor_pred(text:str, model_type='word2vec') -> (np.ndarray):

    '''This function prepares the data to be used in model training. The two components treated here are:
    1. feature: the text from scientific papers that where read out of open access pdfs (large text strings),
    2. target: information on the academic discipline and subdiscipline associated with the papers (labels of type string).
    Both will be transformed while also reducing the number of samples where either feature or target won't nurture desirable results.
    For further elaboration on the whys and hows see jupyter notebook "preproc_and_word2vec_rnn_stef.ipynb" in the notebooks folder.'''


    # Lowercasing everything
    text = text.lower()

    # Removing anything apart from lower case letters
    text = re.sub(r'[^a-z]', ' ', str(text))

    # Removing anything that comes before the abstract
    text= re.sub(r"^.+?(?=abstract)", "", str(text))

    # Tokenizing
    max_length_of_padding = 1000 # to save computational costs, we will implement a longest_token_size here
    num_of_words_to_keep = int(max_length_of_padding*5) # adding some margin, because in spotword removal and word2vec embedding some will be removed
    data = ' '.join(text.split()[:num_of_words_to_keep])
    data = word_tokenize(data)

    # remove single letter words
    data = [word for word in data if len(word)>1]

    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    data = [word for word in data if not word in stop_words]

    if model_type == 'word2vec':
        # Downloading a pre-trained Word2Vec model, that delivers a 50 space vector representation of any word present in the Wikipedia in 2014
        word2vec_transfer = gen_api.load("glove-wiki-gigaword-50")

        # Function to convert a paper (list of words) into a matrix representing the words in the embedding space
        def embed_paper(word2vec_space, paper):
            embedded_paper = []
            for word in paper:
                if word in word2vec_space:
                    embedded_paper.append(word2vec_space[word])
            return np.array(embedded_paper)

        # Function to convert a list of papers into a list of matrices
        def embedding(word2vec_space, series_of_papers):
            embed = []
            for ele in series_of_papers:
                embedded_article = embed_paper(word2vec_space, ele)
                embed.append(embedded_article)
            return embed

        # Adding a column with embedded words from the papers into data frame
        series_embedded = embedding(word2vec_transfer, data)
        data_embedded = series_embedded

        # Padding all the embedded words
        X_pad = pad_sequences(data_embedded, dtype='float32', padding='post', value=0, maxlen=max_length_of_padding)

        return X_pad

    return None
