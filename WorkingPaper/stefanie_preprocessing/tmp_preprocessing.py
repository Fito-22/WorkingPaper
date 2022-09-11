import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import OneHotEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import gensim.downloader as api
from tensorflow.keras.preprocessing.sequence import pad_sequences
import string
from nltk.stem import WordNetLemmatizer

from WorkingPaper.preprocessing.utils import make_broader_subdisciplines_physics, make_broader_subdisciplines_biology

def preprocessor(model_type:str ='embedding') -> (np.ndarray, pd.DataFrame):

    '''This function prepares the data to be used in model training. The two components treated here are:
    1. feature: the text from scientific papers that where read out of open access pdfs (large text strings),
    2. target: information on the academic discipline and subdiscipline associated with the papers (labels of type string).
    Both will be transformed while also reducing the number of samples where either feature or target won't nurture desirable results.
    For further elaboration on the whys and hows see jupyter notebook "preproc_and_word2vec_rnn_stef.ipynb" in the notebooks folder.'''

    # Reading OpenAlex Data+
    openalex_data = pd.read_csv('raw_data/data_3k.csv')
    openalex_data = openalex_data.drop(columns='Unnamed: 0')

    # Reading Arxiv Data
    arxiv_data = pd.read_csv('raw_data/small_dataset.csv')
    arxiv_data = arxiv_data.drop(columns='Unnamed: 0')
    arxiv_data = arxiv_data.rename(columns={'0': 'text'})

    # Turning subtopics into topics
    physics_list = ['astrophysics', 'condensed matter', 'general relativity and quantum cosmology', 'high energy physics - experiment',
                    'high energy physics - lattice', 'high energy physics - phenomenology', 'high energy physics - Tteory',
                    'mathematical physics', 'nonlinear sciences', 'nuclear experiment', 'nuclear theory', 'physics', 'quantum physics']
    arxiv_data['topic'] = arxiv_data['subtopic'].apply(lambda x: 'physics' if x.lower() in physics_list else 'mathematics')

    # Concatenating both datasets
    data = pd.concat([openalex_data, arxiv_data], ignore_index=True)

    # Dropping duplicates
    data.drop_duplicates(subset=['id'], keep='first', inplace=True, ignore_index=True)

    # Lowercasing everything
    data = data.apply(lambda x: x.astype(str).str.lower())

    # Removing anything apart from lower case letters
    data['text'] = data['text'].apply(lambda x: re.sub(r'[^a-z]', ' ', str(x)))

    # Removing anything that comes before the abstract
    data['text'] = data['text'].apply(lambda x: re.sub(r"^.+?(?=abstract)", "", str(x)))

    # Tokenizing
    max_length_of_padding = 1000 # to save computational costs, we will implement a longest_token_size here
    num_of_words_to_keep = int(max_length_of_padding*5) # adding some margin, because in spotword removal and word2vec embedding some will be removed
    data['modified_text'] = data['text'].apply(lambda text: ' '.join(text.split()[:num_of_words_to_keep]))
    data['modified_text'] = data['modified_text'].apply(word_tokenize)

    # Lemmatizing
    data['modified_text'] = data['modified_text'].apply(lambda x: [WordNetLemmatizer().lemmatize(word, pos = 'v') for word in x])
    data['modified_text'] = data['modified_text'].apply(lambda x: [WordNetLemmatizer().lemmatize(word, pos = 'n') for word in x])

    # Removing single letter words
    data['modified_text'] = data['modified_text'].apply(lambda x: [word for word in x if len(word)>1])

    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    data['modified_text'] = data['modified_text'].apply(lambda x: [word for word in x if not word in stop_words])

    # Adding columns with word counts (was used for exploration and is kept here for purpose of consistency)
    data['words_per_modified_text'] = data['modified_text'].apply(lambda x : len(x))

    # Excluding representations with less than 10 tokens
    data = data[data['words_per_modified_text'] > 9].reset_index().drop(columns=['index'], axis=1)

    # Mapping the subtopics from the mapping dict physics to the broader subtopics and exclude 'mathematical physics'
    data = make_broader_subdisciplines_physics(data)
    data = data[data['broader_subtopic'] != 'mathematical physics']

    # Mapping the subtopics from the mapping dict biology to the broader subtopics and exclude 'physiology' and 'computational biology'
    data = make_broader_subdisciplines_biology(data)
    data = data[data['broader_subtopic'] != 'physiology']
    data = data[data['broader_subtopic'] != 'computational biology']

    # Filtering for those topics that occurr more commonly in our data        <----- percentile can be adjusted
    common_topics = (data['topic'].value_counts() > np.percentile(data['topic'].value_counts(), 80)) # topic occurrence until 25th percentile
    filtered_topics = common_topics[common_topics == True].index

    # Filtering for those subtopics that occurr more commonly in our data     <----- instead of cutting of at the mean, can also be adjusted like above
    # common_subtopics = (data['subtopic'].value_counts() > data['subtopic'].value_counts().mean())
    # filtered_subtopics = common_subtopics[common_subtopics == True].index

    # Filtering data according to the topics that are more common
    data = data[data['topic'].isin(list(filtered_topics))]

    # Determining number of topics left for creating the embedding rnn model
    num_of_topics = common_topics.value_counts()[1]

    if model_type == 'word2vec':
        # Downloading a pre-trained Word2Vec model, that delivers a 50 space vector representation of any word present in the Wikipedia in 2014
        word2vec_transfer = api.load("glove-wiki-gigaword-50")

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
        series_embedded = embedding(word2vec_transfer, data['modified_text'])
        data['embedded_text'] = series_embedded

        # Padding all the embedded words
        X_pad = pad_sequences(series_embedded, dtype='float32', padding='post', value=0, maxlen=max_length_of_padding)

    # OneHot Encoding topics
    enc = OneHotEncoder(handle_unknown='ignore', sparse=False)
    topic_targets_enc = pd.DataFrame(enc.fit_transform(data[['topic']]))
    topic_targets_enc.columns = enc.get_feature_names_out()

    print('Index and columns:\n')
    for i, column in enumerate(topic_targets_enc.columns):
        print(i, column)
    print('\n')
    # return print(X_pad, topic_targets_enc)
    if model_type == 'word2vec':
        return (X_pad, topic_targets_enc)
    elif model_type == 'embedding':
        return (data['modified_text'], topic_targets_enc, max_length_of_padding, num_of_topics)

    # Saving preprocessed data in a csv
    data.to_csv(f'../raw_data/preprocessed_data.csv', index=False)

preprocessor(model_type='word2vec')
