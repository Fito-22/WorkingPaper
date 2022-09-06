import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, Sequential, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from WorkingPaper.preprocessing.preprocessing import preprocessor

def train_word2vec_rnn_model(path):

    '''This function traines a RNN model with vectorized tokens already pretrained with Wikipedia data.
    '''

    # Getting the preprocessed data in form of a tuple: (X_pad, topic_targets_enc) and unpacking it
    X_pad, topic_targets_enc = preprocessor(path)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X_pad, topic_targets_enc, test_size=0.3)

    # Weighting the labels as a rebalancing technique
    y_integers = np.argmax(np.array(y_train), axis=1)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_integers), y=y_integers)
    d_class_weights = dict(enumerate(class_weights))

    # Defining Ridge Regularization
    #reg_l2 = regularizers.L2(0.001)

    # Implementing learning rate optimization for Adam
    lr_schedule = ExponentialDecay(
        initial_learning_rate=1e-2,
        decay_steps=10000,
        decay_rate=0.9)

    optimized_adam = Adam(learning_rate=lr_schedule)

    # Model architecture
    def init_word2vev_rnn_model():
        word2vev_rnn_model = Sequential()
        word2vev_rnn_model.add(layers.Masking(input_shape=(X_train.shape[1], X_train.shape[2])))
        word2vev_rnn_model.add(layers.LSTM(20, activation='tanh'))
        #word2vev_rnn_model.add(layers.Dense(20, activation='relu', kernel_regularizer=reg_l2))
        word2vev_rnn_model.add(layers.Dense(15, activation='relu'))
        #word2vev_rnn_model.add(layers.Dense(15, activation='relu', kernel_regularizer=reg_l2))
        word2vev_rnn_model.add(layers.Dense(y_train.shape[1], activation='softmax'))

        word2vev_rnn_model.compile(loss='categorical_crossentropy',
                    #optimizer='adam',
                    optimizer=optimized_adam,
                    metrics=['accuracy'])

        return word2vev_rnn_model

    # Building the model
    word2vev_rnn_model = init_word2vev_rnn_model()

    # Training the model
    es = EarlyStopping(patience=5, restore_best_weights=True)

    history = word2vev_rnn_model.fit(X_train, y_train,
                        class_weight=d_class_weights,
                        batch_size=32,
                        epochs=100,
                        validation_split=0.3,
                        callbacks=[es])

    # Model evaluation with X_test
    res = word2vev_rnn_model.evaluate(X_test, y_test, verbose=0)
    print(f'The accuracy evaluated on the test of this model is {res[1]*100:.3f}%')


    #return print(f'model {word2vev_rnn_model} was created')      <--- for testing purposes
    return word2vev_rnn_model

train_word2vec_rnn_model('raw_data/data_1k.csv')
