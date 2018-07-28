import numpy as np

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
    top_items = sorted(token_to_count.items(),
        key=lambda x: x[1], reverse=True)[:num_tokens]
    return dict(top_items)

def remove_rare_tokens(token_to_count, keep_thres):
    return {tk : c for (tk, c) in token_to_count.iteritems() if c >= keep_thres}

def index_tokens(tokens, start_index=0):
    assert start_index >= 0
    num_tokens = len(tokens)
    token_to_index = dict(zip(tokens, range(start_index, num_tokens + start_index)))
    return token_to_index

def preprocess_sentence(sentence, token_to_index,
    unk_idx=-1, bos_idx=-1, eos_idx=-1, bos_times=0, eos_times=0):
    """If no unk_idx is specified and there are tokens that are sentences that
    have tokens that are not in the dictionary, it will throw an exception.
    it is also possible to look at padding using this type of functionality.
    """
    assert bos_idx == -1 or (bos_idx >= 0 and bos_times >= 0)
    assert eos_idx == -1 or (eos_idx >= 0 and eos_times >= 0)
    assert (unk_idx == -1 and all([tk in token_to_index for tk in sentence])) or (unk_idx >= 0)

    proc_sent = []
    # adding the begin of sentence tokens
    if bos_idx != -1:
        proc_sent.extend([bos_idx] * bos_times)

    # preprocessing the sentence
    for tk in sentence:
        if tk in token_to_index:
            idx = token_to_index[tk]
            proc_sent.append(idx)
        else:
            proc_sent.append(unk_idx)

    # adding the end of sentence token
    if eos_idx != -1:
        proc_sent.extend([eos_idx] * eos_times)

    return proc_sent

def tokenize(sequence, token_to_idx, target_length,
        unk_token='_UNK_', bos_token="_BOS_", eos_token="_EOS_", pad_token="_PAD_",
        num_bos=1, num_eos=1):

    tk_seq = []
    if num_bos > 0:
        bos_idx = token_to_idx[bos_token]
        tk_seq.extend([bos_idx] * num_bos)

    for tk in sequence:
        tk_idx = token_to_index[tk] if tk in token_to_idx else token_to_idx[unk_token]
        tk_seq.append(tk_idx)

    if num_eos > 0:
        eos_idx = token_to_idx[eos_token]
        tk_seq.extend([eos_idx] * num_eos)
    return tk_seq

def convert_onehot_to_indices(y_onehot):
    assert len(y_onehot.shape) == 2
    y_idx = np.where(y_onehot > 0.0)[1]
    return y_idx

def convert_indices_to_onehot(y_idx, num_classes):
    assert len(y_idx.shape) == 1
    num_examples = y_idx.shape[0]
    y_one_hot = np.zeros((num_examples, num_classes), dtype='float32')
    y_one_hot[np.arange(num_examples),  y_idx] = 1.0
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
    out_mask = np.ones(n, max_length, dtype='float32')
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
    assert all(in_dim + idx <= out_dim for in_dim, out_dim, idx in zip(x.shape, output_shape, left_corner_pos))
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
        out_x[idx0:idx0 + n0, idx1:idx1 + n1, idx2:idx2 + n2, idx3:idx3 + n3] = x
    elif len(x.shape) == 5:
        idx0, idx1, idx2, idx3, idx4 = left_corner_pos
        n0, n1, n2, n3, n4 = x.shape
        out_x[idx0:idx0 + n0, idx1:idx1 + n1, idx2:idx2 + n2, idx3:idx3 + n3, idx4:idx4 + n4] = x

    return out_x
