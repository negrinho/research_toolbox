import keras
from keras.layers import Input, Embedding, LSTM, Bidirectional
import keras.backend as K

def get_shape(x):
    return K.int_shape(x)

def create_vector_input(dim):
    return Input(shape=(dim,), dtype='float32')

def create_sequence_input(sequence_length):
    return Input(shape=(sequence_length,), dtype='int32')

def create_image_input(height, width, num_channels):
    return Input(shape=(height, width, num_channels), dtype='float32')

def create_float_tensor_input(shape):
    return Input(shape=shape, dtype='float32')

def create_int_tensor_input(shape):
    return Input(shape=shape, dtype='int32')

def create_basic_sequence_model(input, vocab_size,
        embedding_dim, output_dim, bidirectional, return_sequences,
        embeddings=None, initial_embeddings=None, trainable_embeddings=True,
        lstm=None):
    assert embeddings is None or initial_embeddings is None

    if embeddings is None:
        if initial_embeddings is not None:
            embeddings = Embedding(vocab_size, embedding_dim,
                weights=[initial_embeddings], trainable=trainable_embeddings)
        else:
            embeddings = Embedding(vocab_size, embedding_dim,
                trainable=trainable_embeddings)

    if lstm is None:
        if bidirectional:
            units = int(output_dim / 2)
        else:
            units = output_dim

        lstm = LSTM(units, return_sequences=return_sequences)
        if bidirectional:
            lstm = Bidirectional(lstm)

    # creation of the computational graph.
    x = embeddings(input)
    x = lstm(x)

    return {
        "embeddings" : embeddings,
        "lstm" : lstm,
        "input" : input,
        "output" : x
    }
