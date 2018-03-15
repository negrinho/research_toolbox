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

def reverse_mapping(a_to_b):
    b_to_a = dict([(b, a) for (a, b) in a_to_b.iteritems()])
    return b_to_a

def reverse_non_unique_mapping(a_to_b):
    d = {}
    for (x, y) in a_to_b.iteritems():
        if y not in d:
            d[y] = []
        d[y].append(x)
    return d

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

def convert_onehot_to_idx(y_onehot):
    y_idx = np.where(y_onehot > 0.0)[1]
    return y_idx

def convert_idx_to_onehot(y_idx, num_classes):
    num_images = y_idx.shape[0]
    y_one_hot = np.zeros((num_images, num_classes), dtype='float32')
    y_one_hot[np.arange(num_images),  y_idx] = 1.0
    return y_one_hot