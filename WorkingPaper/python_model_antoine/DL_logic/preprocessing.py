import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

def def_X_y(df):
    '''
    Split the data frame into X_train and X_test
    oneHotencode the two target (topics and subtopics)

    INPUT :
    must be a df with the following columns name
    'paper_text'
    'topic'
    'subtopic'

    OUTPUT :
    - train and test text
    - train and test oneHotEncode target 1 ( topic )
    - train and test oneHotEncode target 2 ( subtopic )

    '''
    X = df[['paper_text']]
    y_topic = df[['topic']]
    y_subtopic = df[['subtopic']]

    enc1 = OneHotEncoder(sparse = False, handle_unknown='ignore')
    y_topic_cat = enc1.fit_transform(df[['topic']])
    new_column_names1 = enc1.get_feature_names_out()
    y_topic_cat = pd.DataFrame(y_topic_cat)
    y_topic_cat.columns = new_column_names1

    enc2 = OneHotEncoder(sparse = False, handle_unknown='ignore')
    y_subtopic_cat = enc2.fit_transform(df[['subtopic']])
    new_column_names1 = enc2.get_feature_names_out()
    y_subtopic_cat = pd.DataFrame(y_subtopic_cat)
    y_subtopic_cat.columns = new_column_names1

    aggregated = X.join(y_topic_cat)
    aggregated = aggregated.join(y_subtopic_cat)

    X_train, X_test = train_test_split(aggregated, test_size=0.33)

    topic_col_name = []
    subtopic_col_name = []
    for elem in X_train.columns:
        if elem[0] == 't':
            topic_col_name.append(elem)
        elif elem[0] == 's':
            subtopic_col_name.append(elem)

    y_topic_train = X_train[topic_col_name]
    y_subtopic_train = X_train[subtopic_col_name]
    y_topic_test = X_test[topic_col_name]
    y_subtopic_test = X_test[subtopic_col_name]

    X_train = X_train[['paper_text']]
    X_test = X_test[['paper_text']]

    return X_train,X_test,y_topic_train,y_subtopic_train,y_topic_test,y_subtopic_test
