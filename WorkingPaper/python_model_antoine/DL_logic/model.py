from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, Dense, Flatten


def model_conv(X_train,X_test,y_train_target):
    '''
    Buiding the model

    INPUT :
    -the train dataset cleaned (dataframe of one column with the text of the papers)

    OUTPUT :
    -the model
    -the X_train_vect that is needed in train.py to fit
    '''
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    ## Vectorizing data
    X_train_vect = pad_sequences(tokenizer.texts_to_sequences(X_train), padding="post", value=0.)
    X_test_vect = pad_sequences(tokenizer.texts_to_sequences(X_test), padding="post", value=0.)
    if X_test_vect.shape[1]>X_train_vect.shape[1]:
        X_test_vect = X_test_vect[:,0:X_train_vect.shape[1]]
    elif X_test_vect.shape[1]<X_train_vect.shape[1]:
        X_train_vect = X_train_vect[:,0:X_test_vect.shape[1]]

    embed_len = 50

    # Conv1D
    cnn = Sequential([
        Embedding(input_dim=len(tokenizer.word_index)+1, input_length=X_train_vect.shape[1], output_dim=embed_len, mask_zero=True),
        Conv1D(20, kernel_size=3),
        Flatten(),
        Dense(y_train_target.shape[1], activation="softmax")
    ])

    cnn.summary()
    cnn.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return cnn, X_train_vect, X_test_vect

