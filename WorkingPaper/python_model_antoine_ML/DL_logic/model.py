from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, Dense, Flatten

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


def model_conv(X_train,X_test,y_train_target):
    '''
    Buiding the model

    INPUT :
    -the train dataset cleaned (dataframe of one column with the text of the papers)

    OUTPUT :
    -the model
    -the X_train_vect that is needed in train.py to fit
    '''

    # Pipeline vectorizer + Naive Bayes
    pipeline_naive_bayes = make_pipeline(CountVectorizer(),
                                     MultinomialNB())
    cv_results = cross_validate(pipeline_naive_bayes, X_train, y_train_target, cv = 5, scoring = ["recall"])
    average_accuracy = cv_results["test_recall"].mean()

    #train_count_vect = count_vect.fit_transform(X_train)
    #test_count_vect = count_vect.transform(X_test)
    ## Vectorizing data

    #if test_count_vect.shape[1]>train_count_vect.shape[1]:
    #    test_count_vect = test_count_vect[:,0:train_count_vect.shape[1]]
    #elif test_count_vect.shape[1]<train_count_vect.shape[1]:
    #    train_count_vect = train_count_vect[:,0:test_count_vect.shape[1]]




    return pipeline_naive_bayes, average_accuracy
