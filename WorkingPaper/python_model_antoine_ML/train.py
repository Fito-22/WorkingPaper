import pandas as pd
import numpy as np

from WorkingPaper.python_model_antoine_ML.load import load
from WorkingPaper.python_model_antoine_ML.DL_logic.cleaning import final_cleaning
from WorkingPaper.python_model_antoine_ML.DL_logic.model import model_conv
from WorkingPaper.python_model_antoine_ML.DL_logic.preprocessing import def_X_y
#from WorkingPaper.save_load_models.save_model import save_model
#from WorkingPaper.save_load_models.load_model import load_model

def training(path = 'raw_data/data_1k.csv'):

    '''
    Train the model with the from the raw data
    split into test and train then evaluate over
    the test set

    INPUT :
    path to the raw data if nothing is given then
    it uses the "small_dataset.csv"

    OUTPUT :
    -the history of the fitted model
    -the model
    -the result of the evaluation on the test set
    '''

    df = load(path)
    print("Loading of the dataset completed")
    #print(df.head(2))
    #print(df['paper_text'])

    df['paper_text'] = df['paper_text'].apply(lambda x: final_cleaning(str(x)))
    print("Cleaning of the Data completed")
    df = df.reset_index().drop(columns='Index')


    X_train,X_test,y_topic_train,y_subtopic_train,y_topic_test,y_subtopic_test = def_X_y(df)
    print("Preprocessing of target completed")
    print("Train test split completed")
    print(X_train.head())
    print(X_train.shape)
    print(y_topic_train['topic_Biology'])
    print(y_topic_train.shape)
    #print(y_topic_test.head())
    #print(y_topic_test.shape)


    pipeline,cross_score = model_conv(X_train['paper_text'],X_test['paper_text'],y_topic_train)
    #print(f'X_train_vect_shape : {X_train_vect.shape}')
    #print(f'y_train_topic_shape : {y_topic_train.shape}')
    #print(f'X_test_vect_shape : {X_test_vect.shape}')
    #print(X_train_vect)


    #history = cnn_model.fit(X_train_vect, y_topic_train, validation_split=0.3,epochs=3, batch_size=30, verbose=1)
    print('Model fitted to the train set')

    print('score : ')
    print(cross_score)


    #ld_model = load_model()

    #eval_test = cnn_model.evaluate(X_test_vect,y_topic_test)
    #print('Model evaluated ont the test set')
    #print(eval_test)
    #test = cnn_model.predict(X_test)
    #print(test)




    return None



if __name__ == '__main__':
    training()
