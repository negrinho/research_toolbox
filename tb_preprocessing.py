### for simple NLP
def keep_short_sentences(sents, maxlen):
    return [s for s in sents if len(s) <= maxlen]
# TODO: change this for sequences maybe. simpler.

def count_tokens(sents):
    tk_to_cnt = {}
    for s in sents:
        for tk in s:
            if tk in tk_to_cnt:
                tk_to_cnt[tk] += 1
            else:
                tk_to_cnt[tk] = 1
    return tk_to_cnt

def keep_most_frequent_tokens(tk_to_cnt, num_tokens):
    top_items = sorted(tk_to_cnt.items(), 
        key=lambda x: x[1], reverse=True)[:num_tokens]
    return dict(top_items)

def remove_rare_tokens(tk_to_cnt, keep_thres):
    return {tk : c for (tk, c) in tk_to_cnt.iteritems() if c >= keep_thres}

def index_tokens(tokens, start_index=0):
    assert start_index >= 0
    num_tokens = len(tokens)
    tk_to_idx = dict(zip(tokens, range(start_index, num_tokens + start_index)))
    return tk_to_idx

def reverse_mapping(a_to_b):
    b_to_a = dict([(b, a) for (a, b) in a_to_b.iteritems()])
    return b_to_a

def reverse_nonunique_mapping(a_to_b):
    d = {}
    for (x, y) in a_to_b.iteritems():
        if y not in d:
            d[y] = []
        d[y].append(x)
    return d

def preprocess_sentence(sent, tk_to_idx, 
    unk_idx=-1, bos_idx=-1, eos_idx=-1, 
    bos_times=0, eos_times=0):
    """If no unk_idx is specified and there are tokens that are sentences that 
    have tokens that are not in the dictionary, it will throw an exception.
    it is also possible to look at padding using this type of functionality.
    """
    assert bos_idx == -1 or (bos_idx >= 0 and bos_times >= 0)  
    assert eos_idx == -1 or (eos_idx >= 0 and eos_times >= 0)  
    assert (unk_idx == -1 and all([tk in tk_to_idx for tk in sent])) or (unk_idx >= 0)

    proc_sent = []
    # adding the begin of sentence tokens
    if bos_idx != -1:
        proc_sent.extend([bos_idx] * bos_times)

    # preprocessing the sentence
    for tk in sent:
        if tk in tk_to_idx:
            idx = tk_to_idx[tk]
            proc_sent.append(idx)
        else:
            proc_sent.append(unk_idx)
            
    # adding the end of sentence token
    if eos_idx != -1:
        proc_sent.extend([eos_idx] * eos_times)

    return proc_sent
