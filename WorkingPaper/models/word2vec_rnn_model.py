#from asyncio.constants import SENDFILE_FALLBACK_READBUFFER_SIZE
#from http.cookies import SimpleCookie
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, Sequential, regularizers
from tensorflow.keras.callbacks import EarlyStopping
#from tensorflow.keras.optimizers import Adam
#from tensorflow.keras.optimizers.schedules import ExponentialDecay

#from WorkingPaper.preprocessing.preprocessing import preprocessor
from WorkingPaper.stefanie_preprocessing.tmp_preprocessing import preprocessor
from WorkingPaper.save_load_models.save_model import save_model


def train_word2vec_rnn_model():

    '''This function traines a RNN model with vectorized tokens already pretrained with Wikipedia data.
    '''

    # Getting the preprocessed data in form of a tuple: (X_pad, topic_targets_enc) and unpacking it

    print('Preprocessing the data...')

    X_pad, topic_targets_enc = preprocessor(model_type='word2vec')

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X_pad, topic_targets_enc, test_size=0.3)

    # Weighting the labels as a rebalancing technique
    y_integers = np.argmax(np.array(y_train), axis=1)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_integers), y=y_integers)
    d_class_weights = dict(enumerate(class_weights))

    # Defining Ridge Regularization
    reg_l2 = regularizers.L2(0.001)

    # Implementing learning rate optimization for Adam
#    lr_schedule = ExponentialDecay(
#        initial_learning_rate=1e-2,
#        decay_steps=10000,
#        decay_rate=0.9)

#    optimized_adam = Adam(learning_rate=lr_schedule)

    # Defining the F1 score manually
#    from keras import backend as K

#    def recall_m(y_true, y_pred):
#        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#        recall = true_positives / (possible_positives + K.epsilon())
#        return recall

#    def precision_m(y_true, y_pred):
#        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#        precision = true_positives / (predicted_positives + K.epsilon())
#        return precision

#    def f1_m(y_true, y_pred):
#        precision = precision_m(y_true, y_pred)
#       recall = recall_m(y_true, y_pred)
#        return 2*((precision*recall)/(precision+recall+K.epsilon()))

    # Model architecture
    def init_word2vev_rnn_model():
        word2vev_rnn_model = Sequential()
        word2vev_rnn_model.add(layers.Masking(input_shape=(X_train.shape[1], X_train.shape[2])))
        word2vev_rnn_model.add(layers.LSTM(20, activation='tanh'))
        word2vev_rnn_model.add(layers.Dense(20, activation='relu', kernel_regularizer=reg_l2))
        word2vev_rnn_model.add(layers.Dense(20, activation='relu'))
        word2vev_rnn_model.add(layers.Dense(15, activation='relu', kernel_regularizer=reg_l2))
        word2vev_rnn_model.add(layers.Dense(y_train.shape[1], activation='softmax'))

        word2vev_rnn_model.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    #optimizer=optimized_adam,
                    #optimizer='rmsprop',
                    metrics=['accuracy'])

        return word2vev_rnn_model

    # Building the model
    print('Building the model...')
    word2vev_rnn_model = init_word2vev_rnn_model()

    # Training the model
    print('Training the model...')
    es = EarlyStopping(patience=5, monitor="val_loss", restore_best_weights=True)

    history = word2vev_rnn_model.fit(X_train, y_train,
                        class_weight=d_class_weights,
                        batch_size=32,
                        epochs=100,
                        validation_split=0.3,
                        callbacks=[es])

    # Model evaluation with X_test
    print('Evaluating...')
    loss, acc = word2vev_rnn_model.evaluate(X_test, y_test, verbose=0)
    print(f'Accuracy: {acc}')

    print('Predicting...')
    X_pred = np.expand_dims(X_test[0,:,:], axis=0)
    print(X_pred.shape)
    y_pred = word2vev_rnn_model.predict(X_pred)
    print(type(y_pred),'\n',y_pred)

    #return print(f'model {word2vev_rnn_model} was created')      <--- for testing purposes

    return word2vev_rnn_model, history

if __name__ == '__main__':
    print('Starting...')
    word2vev_rnn_model, history = train_word2vec_rnn_model()
    params = {
        'optimizer':'adam',
        'embeded_len':'50',
        'batch_size':'32',
        'patience': '5',
        'r2_regularization': '0.001',
        'class_weights': 'balanced'
    }
    metrics={'accuracy':np.max(history.history['accuracy'])}

    save_model(model=word2vev_rnn_model, params=params, metrics=metrics)
