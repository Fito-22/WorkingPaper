from dataclasses import dataclass
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from WorkingPaper.preprocessing.preprocessing import preprocessor

def train_embedding_rnn_model():
    '''
    This Function is used to train an RNN model that uses a custom embeddings layer.
    '''

    # Getting the preprocessed data in form of a tuple: (X_pad, topic_targets_enc) and unpacking it
    data, topic_targets_enc, max_length_of_padding, num_of_topics = preprocessor('raw_data/data_3k.csv', 'embedding')

    # X = data
    # Splitting the data into training and testing datasets
    X_topic_train, X_topic_test, y_topic_train, y_topic_test = train_test_split(data,
                                                                                topic_targets_enc,
                                                                                test_size=0.3,
                                                                                random_state=1)

    # Applying weights to each topic to rebalance the dataset, less occurrence means higher weight
    y_integers = np.argmax(np.array(y_topic_train), axis=1)
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_integers), y=y_integers)
    y_topic_class_weights = dict(enumerate(class_weights))
    y_topic_class_weights

    # Creating a Tokenizer and fitting only the train set on it
    tk = Tokenizer()
    tk.fit_on_texts(X_topic_train)
    sequences = tk.texts_to_sequences(X_topic_train)

    # Padding the data and getting the vocab size for creating the neural network
    topic_vocab_size = len(tk.word_index)
    X_pad_topic = pad_sequences(sequences, dtype='float32', padding='post', maxlen=max_length_of_padding)

    # Defining the model architecture
    def init_model():
        embedding_size = 50 # Size of your embedding space = size of the vector representing each word

        topic_model = Sequential()
        topic_model.add(layers.Embedding(
            input_dim=topic_vocab_size+1,
            output_dim=embedding_size,
            mask_zero=True, # Built-in masking layer :)
        ))

        topic_model.add(layers.LSTM(20))
        topic_model.add(layers.Dense(num_of_topics, activation="softmax"))
        topic_model.summary()

        topic_model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

        return topic_model

    # Initiating the model
    custom_embed_layer_model = init_model()

    # Fitting the model on the train data using class weights
    es = EarlyStopping(patience=5,restore_best_weights=True, monitor='val_accuracy')
    history = custom_embed_layer_model.fit(X_pad_topic, y_topic_train,
                          class_weight=y_topic_class_weights,
                          epochs=20, validation_split=0.3, batch_size=32,
                          verbose=1, callbacks=[es])

    # Padding the test set and evaluating on it
    sequences2 = tk.texts_to_sequences(X_topic_test)
    X_pad_topic2 = pad_sequences(sequences2, dtype='float32', padding='post')

    result = custom_embed_layer_model.evaluate(X_pad_topic2,y_topic_test)

    print(f'The accuracy evaluated on the test of this model is {result[1]*100:.3f}%')

    return custom_embed_layer_model

train_embedding_rnn_model()
