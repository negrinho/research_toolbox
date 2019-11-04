import research_toolbox.tb_utils as tb_ut


def increment(key2cnt, key):
    if key not in key2cnt:
        key2cnt[key] = 1
    else:
        key2cnt[key] += 1
    return key2cnt


def add(key2cnt, key, count):
    if key not in key2cnt:
        key2cnt[key] = count
    else:
        key2cnt[key] += count
    return key2cnt


def increment_with_list(key2cnt, lst):
    for key in lst:
        increment(key2cnt, key)
    return key2cnt


def update_with_dict(key2cnt, from_key2count):
    for key, cnt in from_key2count.items():
        add(key2cnt, key, cnt)
    return key2cnt


def topk(key2cnt, k):
    return tb_ut.sort_dict_items(key2cnt, by_key=False, decreasing=True)[:k]


def bottomk(key2cnt, k):
    return tb_ut.sort_dict_items(key2cnt, by_key=False, decreasing=False)[:k]


def keep_bigger_or_equal_than_threshold(key2cnt, threshold):
    return {key: cnt for (key, cnt) in key2cnt.items() if cnt >= threshold}


def keep_smaller_or_equal_than_threshold(key2cnt, threshold):
    return {key: cnt for (key, cnt) in key2cnt.items() if cnt <= threshold}


def get_count_total(key2cnt):
    return sum(key2cnt.values())


def counts_to_fractions(key2cnt):
    total = float(get_count_total(key2cnt))
    key2frac = {key: cnt / total for (key, cnt) in key2cnt.items()}
    return key2frac


def cummulative_from_list(key_frac_lst):
    acc = 0.0
    out_lst = []
    for (key, frac) in key_frac_lst:
        acc += frac
        out_lst.append((key, acc))
    return out_lst


def fractional_topk(key2cnt, k, cummulative=True):
    key2frac = counts_to_fractions(key2cnt)
    out_lst = topk(key2frac, k)
    if cummulative:
        out_lst = cummulative_from_list(out_lst)
    return out_lst


def fractional_bottomk(key2cnt, k, cummulative=True):
    key2frac = counts_to_fractions(key2cnt)
    out_lst = bottomk(key2frac, k)
    if cummulative:
        out_lst = cummulative_from_list(out_lst)
    return out_lst
