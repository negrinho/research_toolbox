import numpy as np
import re


def get_ch2idx(with_pad=True,
               with_bos=True,
               with_eos=True,
               with_unk=True,
               with_overflow=True):
    ch_lst = []
    if with_unk:
        ch_lst.append("_UNK_")
    if with_bos:
        ch_lst.append("_BOS_")
    if with_eos:
        ch_lst.append("_EOS_")
    if with_pad:
        ch_lst.append("_PAD_")
    if with_overflow:
        ch_lst.append("_OVERFLOW_")
    ch_lst.extend(["\t", "\n"])
    ch_lst.extend([chr(idx) for idx in range(32, 127)])
    ch2idx = dict(list(zip(ch_lst, list(range(len(ch_lst))))))
    return ch2idx


### for simple NLP
def keep_short_sentences(sentence_lst, max_length):
    return [s for s in sentence_lst if len(s) <= max_length]


def count_tokens(sentence_lst):
    token_to_count = {}
    for s in sentence_lst:
        for tk in s:
            if tk in token_to_count:
                token_to_count[tk] += 1
            else:
                token_to_count[tk] = 1
    return token_to_count


def keep_most_frequent_tokens(token_to_count, num_tokens):
    top_items = sorted(
        list(token_to_count.items()), key=lambda x: x[1], reverse=True)[:num_tokens]
    return dict(top_items)


def remove_rare_tokens(token_to_count, keep_thres):
    return {tk: c for (tk, c) in token_to_count.items() if c >= keep_thres}


def index_tokens(tokens, start_index=0):
    assert start_index >= 0
    num_tokens = len(tokens)
    token_to_index = dict(
        list(zip(tokens, list(range(start_index, num_tokens + start_index)))))
    return token_to_index


def index_with_special_tokens(tokens,
                              with_pad=True,
                              with_bos=True,
                              with_eos=True,
                              with_unk=True,
                              with_overflow=True,
                              pad_token="_PAD_",
                              bos_token="_BOS_",
                              eos_token="_EOS_",
                              unk_token="_UNK_",
                              overflow_token="_OVERFLOW_"):

    num_normal_tokens = len(tokens)
    num_special_tokens = sum(
        [with_pad, with_bos, with_eos, with_unk, with_overflow])
    token_to_index = dict(
        list(zip(tokens,
            list(range(num_special_tokens, num_normal_tokens + num_special_tokens)))))

    idx = 0
    if with_pad:
        assert pad_token not in token_to_index
        token_to_index[pad_token] = idx
        idx += 1
    if with_bos:
        assert bos_token not in token_to_index
        token_to_index[bos_token] = idx
        idx += 1
    if with_eos:
        assert eos_token not in token_to_index
        token_to_index[eos_token] = idx
        idx += 1
    if with_unk:
        assert unk_token not in token_to_index
        token_to_index[unk_token] = idx
        idx += 1
    if with_overflow:
        assert overflow_token not in token_to_index
        token_to_index[overflow_token] = idx
        idx += 1
    return token_to_index


def tokenize(sequence,
             token_to_index,
             target_length,
             unk_token='_UNK_',
             bos_token="_BOS_",
             eos_token="_EOS_",
             pad_token="_PAD_",
             overflow_token="_OVERFLOW_",
             num_bos=1,
             num_eos=1):

    out_seq = []
    if num_bos > 0:
        bos_idx = token_to_index[bos_token]
        out_seq.extend([bos_idx] * num_bos)

    for tk in sequence:
        tk_idx = token_to_index[tk] if tk in token_to_index else token_to_index[
            unk_token]
        out_seq.append(tk_idx)

    if num_eos > 0:
        eos_idx = token_to_index[eos_token]
        out_seq.extend([eos_idx] * num_eos)

    if len(out_seq) <= target_length:
        pad_length = target_length - len(out_seq)
        pad_idx = token_to_index[pad_token]
        out_seq.extend([pad_idx] * pad_length)
    else:
        overflow_idx = token_to_index[overflow_token]
        out_seq[target_length - 1] = overflow_idx
        out_seq = out_seq[:target_length]
    return out_seq


def untokenize(sequence, index_to_token):
    return [index_to_token[idx] for idx in sequence]


def character_tokenize(sentence,
                       char_to_index,
                       target_length,
                       unk_token='_UNK_',
                       bos_token="_BOS_",
                       eos_token="_EOS_",
                       pad_token="_PAD_",
                       num_bos=1,
                       num_eos=1):
    return tokenize(
        sentence,
        char_to_index,
        target_length,
        unk_token=unk_token,
        bos_token=bos_token,
        eos_token=eos_token,
        pad_token=pad_token,
        num_bos=num_bos,
        num_eos=num_eos)


def word_tokenize(sentence,
                  word_to_index,
                  target_length,
                  unk_token='_UNK_',
                  bos_token="_BOS_",
                  eos_token="_EOS_",
                  pad_token="_PAD_",
                  num_bos=1,
                  num_eos=1):
    tokens = split_sentence_into_word_tokens(sentence)
    return tokenize(
        tokens,
        word_to_index,
        target_length,
        unk_token=unk_token,
        bos_token=bos_token,
        eos_token=eos_token,
        pad_token=pad_token,
        num_bos=num_bos,
        num_eos=num_eos)


def zero_out_digits(s):
    out_s = []
    for ch in s:
        if ch.isdigit():
            out_s.append('0')
        else:
            out_s.append(ch)
    return ''.join(out_s)


def lowercase(s):
    return s.lower()


def split_sentence_into_word_tokens(sentence):
    from nltk import word_tokenize as nltk_word_tokenize
    return nltk_word_tokenize(sentence)


def convert_onehot_to_indices(y_onehot):
    assert len(y_onehot.shape) == 2
    y_idx = np.where(y_onehot > 0.0)[1]
    return y_idx


def convert_indices_to_onehot(y_idx, num_classes):
    assert len(y_idx.shape) == 1
    num_examples = y_idx.shape[0]
    y_one_hot = np.zeros((num_examples, num_classes), dtype='float32')
    y_one_hot[np.arange(num_examples), y_idx] = 1.0
    return y_one_hot


def all_mask(shape):
    return np.ones(shape, dtype='float32')


def none_mask(shape):
    return np.zeros(shape, dtype='float32')


def mask_union(mask_lst):
    out_mask = np.zeros_like(mask_lst[0], dtype='float32')
    for m in mask_lst:
        out_mask += m
    out_mask = np.clip(out_mask, 0.0, 1.0)
    return out_mask


def mask_intersection(mask_lst):
    out_mask = np.ones_like(mask_lst[0], dtype='float32')
    for m in mask_lst:
        assert m.shape == out_mask.shape
        out_mask *= m
    return out_mask


def mask_invert(mask):
    return 1.0 - mask


def mask_from_lengths(length_lst, max_length):
    n = len(length_lst)
    out_mask = np.ones((n, max_length), dtype='float32')
    for i, x in enumerate(length_lst):
        out_mask[i, x:] = 0.0
    return out_mask


def mask_indices1d(mask):
    assert len(mask.shape) == 1
    return np.where(mask > 0.5)[0]


def true_indices1d(x):
    assert len(x.shape) == 1
    return np.where(x)[0]


def num_dims(x):
    return len(x.shape)


def pad_tensor(x, output_shape, pad_val, left_corner_pos):
    assert len(x.shape) == len(output_shape)
    assert len(left_corner_pos) == len(output_shape)
    assert all(
        in_dim + idx <= out_dim
        for in_dim, out_dim, idx in zip(x.shape, output_shape, left_corner_pos))
    # NOTE: current limitation, but not a big problem in practice. improve later.
    assert len(output_shape) <= 5 and len(output_shape) > 0
    assert type(output_shape) == tuple and type(left_corner_pos) == tuple

    out_x = np.empty(output_shape, x.dtype)
    out_x.fill(pad_val)

    if len(x.shape) == 1:
        idx0 = left_corner_pos[0]
        n0 = x.shape[0]
        out_x[idx0:idx0 + n0] = x
    elif len(x.shape) == 2:
        idx0, idx1 = left_corner_pos
        n0, n1 = x.shape
        out_x[idx0:idx0 + n0, idx1:idx1 + n1] = x
    elif len(x.shape) == 3:
        idx0, idx1, idx2 = left_corner_pos
        n0, n1, n2 = x.shape
        out_x[idx0:idx0 + n0, idx1:idx1 + n1, idx2:idx2 + n2] = x
    elif len(x.shape) == 4:
        idx0, idx1, idx2, idx3 = left_corner_pos
        n0, n1, n2, n3 = x.shape
        out_x[idx0:idx0 + n0, idx1:idx1 + n1, idx2:idx2 + n2, idx3:idx3 +
              n3] = x
    elif len(x.shape) == 5:
        idx0, idx1, idx2, idx3, idx4 = left_corner_pos
        n0, n1, n2, n3, n4 = x.shape
        out_x[idx0:idx0 + n0, idx1:idx1 + n1, idx2:idx2 + n2, idx3:idx3 +
              n3, idx4:idx4 + n4] = x
    return out_x


def reshape_apply(x, reshape_fn, apply_fn, pre_shape=None, post_shape=None):
    if pre_shape is not None:
        x = reshape_fn(x, pre_shape)
    x = apply_fn(x)
    if post_shape is not None:
        x = reshape_fn(x, post_shape)
    return x


def pack(xs):
    return np.concatenate(xs)


def unpack(x):
    return np.split(x, x.shape[0])


def multi_to_flat_indices(indices, shape):
    assert len(indices.shape) == 2
    assert indices.shape[1] == len(shape)
    aux = shape[1:] + (1,)
    strides = np.cumprod(aux[::-1])[::-1].astype('int64')
    flat_indices = (indices * strides[None, :]).sum(axis=1)
    return flat_indices


def flat_to_multi_indices(indices, shape):
    assert len(indices.shape) == 1
    num_indices = indices.shape[0]
    num_dims = len(shape)
    aux = shape[1:] + (1,)
    strides = np.cumprod(aux[::-1])[::-1].astype('int64')
    multi_indices = np.zeros((num_indices, num_dims), dtype='int64')
    acc = np.array(indices)
    for idx in range(num_dims):
        x = strides[idx]
        multi_indices[:, idx] = np.floor_divide(acc, x)
        acc -= x * multi_indices[:, idx]
    return multi_indices


def gather(x, indices):
    assert len(x.shape) == 2
    assert len(indices.shape) == 1
    return x[indices]


# NOTE: this is a stateful update.
def scatter_update(x, y, from_indices, to_indices):
    assert len(to_indices.shape) == 1
    assert len(from_indices.shape) == 1
    assert from_indices.shape[0] == to_indices.shape[0]
    y[to_indices] = x[from_indices]
    return y


def sparse_column_to_multi_indices(col_indices_lst):
    num_indices = sum([len(col_idxs) for col_idxs in col_indices_lst])
    out = np.zeros((num_indices, 2), dtype='int64')
    left = 0
    for row_idx, col_idxs in enumerate(col_indices_lst):
        right = left + len(col_idxs)
        out[left:right, 0] = row_idx
        out[left:right, 1] = col_idxs
        left = right
    return out


def sparse_column_with_row_to_multi_indices(row_indices, col_indices_lst):
    assert len(row_indices) == len(col_indices_lst)
    num_indices = sum([len(col_idxs) for col_idxs in col_indices_lst])
    out = np.zeros((num_indices, 2), dtype='int64')
    left = 0
    for idx in enumerate(col_indices_lst):
        col_idxs = col_indices_lst[idx]
        right = left + len(col_idxs)
        out[left:right, 0] = row_indices[idx]
        out[left:right, 1] = col_idxs
        left = right
    return out


def sorting_indices(x, decreasing):
    assert len(x.shape) == 2
    indices = np.argsort(x, axis=1)
    if decreasing:
        indices = np.fliplr(indices)
    return indices


def topk(x, k):
    indices = sorting_indices(x, True)[:, :k]
    aux_indices = np.tile(np.arange(x.shape[0])[:, None], (1, k))
    return x[aux_indices, indices]


def bottomk(x, k):
    indices = sorting_indices(x, False)[:, :k]
    aux_indices = np.tile(np.arange(x.shape[0])[:, None], (1, k))
    return x[aux_indices, indices]


def reflect(x, axis):
    return np.flip(x, axis)
