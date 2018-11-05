import keras
from keras.layers import Input, Embedding, LSTM, Bidirectional, Reshape, Lambda
from keras.layers import concatenate
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


def reshape_with_batch_dimension(x, output_shape):
    return Lambda(lambda y: K.reshape(y, output_shape))(x)


def reshape_without_batch_dimension(x, output_shape):
    return Reshape(output_shape)(x)


def flatten_all_but_last_dimension(x):
    shape = get_shape(x)
    return reshape_with_batch_dimension(x, (-1, shape[-1]))


def concatenate_along_last_axis(lst):
    assert len(lst) > 0
    if len(lst) > 2:
        return concatenate(lst)
    else:
        return lst[0]


def create_basic_sequence_model(input,
                                sequence_length,
                                vocab_size,
                                embedding_dim,
                                output_dim,
                                return_sequences,
                                bidirectional=True,
                                embeddings=None,
                                initial_embeddings=None,
                                trainable_embeddings=True,
                                lstm=None):
    assert embeddings is None or initial_embeddings is None

    if embeddings is None:
        if initial_embeddings is not None:
            embeddings = Embedding(
                vocab_size,
                embedding_dim,
                weights=[initial_embeddings],
                trainable=trainable_embeddings)
        else:
            embeddings = Embedding(
                vocab_size, embedding_dim, trainable=trainable_embeddings)

    if lstm is None:
        units = int(output_dim / 2) if bidirectional else output_dim
        lstm = LSTM(units, return_sequences=return_sequences)
        if bidirectional:
            lstm = Bidirectional(lstm)

    # creation of the computational graph.
    emb_sequence = embeddings(input)
    output = lstm(emb_sequence)

    return {
        "model": {
            "embeddings": embeddings,
            "lstm": lstm,
        },
        "nodes": {
            "input": input,
            "embedded_sequence": emb_sequence,
            "output": output
        }
    }


# class DataGenerator(keras.utils.Sequence):

#     def __init__(self):
#         pass

#     # number of batches
#     def __len__(self):
#         pass

#     # get one batch of data: (X, y).
#     def __getitem__(self, index):
#         return (batch_X, batch_y)

#     # for reshuffling indices after the end of the epoch.
#     def on_epoch_end(self):
#         pass

### useful links
# - functional API: https://keras.io/getting-started/functional-api-guide/
# - multiple outputs and multiplee losses: https://www.pyimagesearch.com/2018/06/04/keras-multiple-outputs-and-multiple-losses/