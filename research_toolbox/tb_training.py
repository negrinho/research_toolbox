### sequences, schedules, and counters (useful for training)
import numpy as np
import research_toolbox.tb_io as tb_io
import research_toolbox.tb_filesystem as tb_fs
import research_toolbox.tb_utils as tb_ut


### for storing the data
class InMemoryDataset:

    def __init__(self, X, y, shuffle_at_epoch_begin, batch_transform_fn=None):
        if X.shape[0] != y.shape[0]:
            assert ValueError("X and y the same number of examples.")

        self.X = X
        self.y = y
        self.shuffle_at_epoch_begin = shuffle_at_epoch_begin
        self.batch_transform_fn = batch_transform_fn
        self.iter_i = 0

    def get_num_examples(self):
        return self.X.shape[0]

    def next_batch(self, batch_size):
        n = self.X.shape[0]
        i = self.iter_i

        # shuffling step.
        if i == 0 and self.shuffle_at_epoch_begin:
            inds = np.random.permutation(n)
            self.X = self.X[inds]
            self.y = self.y[inds]

        # getting the batch.
        eff_batch_size = min(batch_size, n - i)
        X_batch = self.X[i:i + eff_batch_size]
        y_batch = self.y[i:i + eff_batch_size]
        self.iter_i = (self.iter_i + eff_batch_size) % n

        # transform if a transform function was defined.
        if self.batch_transform_fn != None:
            X_batch_out, y_batch_out = self.batch_transform_fn(X_batch, y_batch)

        return (X_batch_out, y_batch_out)


class PatienceRateSchedule:

    def __init__(self,
                 rate_init,
                 rate_mult,
                 rate_patience,
                 rate_max=np.inf,
                 rate_min=-np.inf,
                 minimizing=True):
        assert (rate_patience > 0 and (rate_mult > 0.0 and rate_mult <= 1.0) and
                rate_min > 0.0 and rate_max >= rate_min)

        self.rate_init = rate_init
        self.rate_mult = rate_mult
        self.rate_patience = rate_patience
        self.rate_max = rate_max
        self.rate_min = rate_min
        self.minimizing = minimizing

        # for keeping track of the learning rate updates
        self.counter = rate_patience
        self.prev_value = np.inf if minimizing else -np.inf
        self.cur_rate = rate_init

    def update(self, v):
        # if it improved, reset counter.
        if (self.minimizing and v < self.prev_value) or (
            (not self.minimizing) and v > self.prev_value):
            self.counter = self.rate_patience
        else:
            self.counter -= 1
            if self.counter == 0:
                self.cur_rate *= self.rate_mult
                # rate truncation
                self.cur_rate = min(
                    max(self.rate_min, self.cur_rate), self.rate_max)
                self.counter = self.rate_patience

        self.prev_value = min(v, self.prev_value) if self.minimizing else max(
            v, self.prev_value)

    def get_rate(self):
        return self.cur_rate


class AdditiveRateSchedule:

    def __init__(self, rate_init, rate_end, duration):
        assert rate_init > 0 and rate_end > 0 and duration > 0

        self.rate_init = rate_init
        self.rate_delta = (rate_init - rate_end) / float(duration)
        self.duration = duration

        self.num_steps = 0
        self.cur_rate = rate_init

    def update(self, v):
        assert self.num_steps < self.duration
        self.num_steps += 1
        self.cur_rate += self.rate_delta

    def get_rate(self):
        return self.cur_rate


class MultiplicativeRateSchedule:

    def __init__(self, rate_init, rate_mult):
        assert rate_init > 0 and rate_mult > 0

        self.rate_init = rate_init
        self.rate_mult = rate_mult
        self.cur_rate = rate_init

    def update(self, v):
        self.cur_rate *= self.rate_mult

    def get_rate(self):
        return self.cur_rate


class ConstantRateSchedule:

    def __init__(self, rate):
        self.rate = rate

    def update(self, v):
        pass

    def get_rate(self):
        return self.rate


class StepwiseRateSchedule:

    def __init__(self, rates, durations):
        assert len(rates) == len(durations)

        self.schedule = PiecewiseSchedule(
            [ConstantRateSchedule(r) for r in rates], durations)

    def update(self, v):
        self.schedule.update(v)

    def get_rate(self):
        return self.schedule.get_rate()


class PiecewiseSchedule:

    def __init__(self, schedules, durations):
        assert len(schedules) > 0 and len(schedules) == len(durations) and (all(
            [d > 0 for d in durations]))

        self.schedules = schedules
        self.durations = durations

        self.num_steps = 0
        self.idx = 0

    def update(self, v):
        self.num_steps += 1
        n = self.num_steps
        self.idx = 0
        for d in self.durations:
            n -= d
            if n < 0:
                break
            self.idx += 1

        self.schedules[self.idx].update(v)

    def get_rate(self):
        return self.schedules[self.idx].get_rate()


class CosineRateSchedule:

    def __init__(self, rate_start, rate_end, duration):
        assert duration > 0

        self.rate_start = rate_start
        self.rate_end = rate_end
        self.duration = duration
        self.num_steps = 0

    def update(self, v):
        assert self.num_steps < self.duration
        self.num_steps += 1

    def get_rate(self):
        return self.rate_end + 0.5 * (self.rate_start - self.rate_end) * (
            1 + np.cos(float(self.num_steps) / self.duration * np.pi))


class PatienceCounter:

    def __init__(self,
                 patience,
                 init_val=None,
                 minimizing=True,
                 improv_thres=0.0):
        assert patience > 0

        self.minimizing = minimizing
        self.improv_thres = improv_thres
        self.patience = patience
        self.counter = patience

        if init_val is not None:
            self.best = init_val
        else:
            if minimizing:
                self.best = np.inf
            else:
                self.best = -np.inf

    def update(self, v):
        assert self.counter > 0

        # if it improved, reset counter.
        if (self.minimizing and self.best - v > self.improv_thres) or (
            (not self.minimizing) and v - self.best > self.improv_thres):
            self.counter = self.patience
        else:
            self.counter -= 1

        # update with the best seen so far.
        if self.minimizing:
            self.best = min(v, self.best)
        else:
            self.best = max(v, self.best)

    def has_stopped(self):
        return self.counter == 0


def get_random_step_schedule(min_rate_power,
                             max_rate_power,
                             min_duration_power,
                             max_duration_power,
                             num_switch_points,
                             ensure_decreasing_rate=False,
                             ensure_increasing_rate=False,
                             ensure_decreasing_duration=False,
                             ensure_increasing_duration=False,
                             is_int_type=False):
    assert min_duration_power >= 0

    def sample(possible_values, num_samples, ensure_decreasing,
               ensure_increasing):
        assert not (ensure_decreasing and ensure_increasing)

        if ensure_decreasing:
            possible_values = possible_values[::-1]

        if ensure_decreasing or ensure_increasing:
            values = []
            idx = 0
            for _ in range(num_switch_points):
                idx = np.random.randint(idx, len(possible_values))
                values.append(possible_values[idx])
            values = np.array(values)
        else:
            values = np.random.choice(possible_values, num_switch_points)
        return values

    possible_rates = tb_ut.powers_of_two(
        min_rate_power, max_rate_power, is_int_type=is_int_type)
    rates = sample(possible_rates, num_switch_points, ensure_decreasing_rate,
                   ensure_increasing_rate)

    possible_durations = tb_ut.powers_of_two(
        min_duration_power, max_duration_power, is_int_type=True)
    durations = sample(possible_durations, num_switch_points,
                       ensure_decreasing_duration, ensure_increasing_duration)

    return StepwiseRateSchedule(rates, durations), rates, durations


# NOTE: actually, state_dict needs a copy.
# example
# cond_fn = lambda old_x, x: old_x['acc'] < x['acc']
# save_fn = lambda x: tb.copy_update_dict(x, {'model' : x['model'].save_dict()})
# load_fn = lambda x: x,
# for example, but it allows more complex functionality.
class Checkpoint:

    def __init__(self, initial_state, cond_fn, save_fn, load_fn):
        self.state = initial_state
        self.cond_fn = cond_fn
        self.save_fn = save_fn
        self.load_fn = load_fn

    def update(self, x):
        if self.cond_fn(self.state, x):
            self.state = self.save_fn(x)

    def get(self):
        return self.load_fn(self.state)


# TODO: perhaps can be improved to guarantee that all elements were saved
# correctly, e.g., right now it can timeout and get back a corrupted file.
# maintain versions with dates (most recent is best).
# TODO: perhaps later add more functions for removing only certain ones.
# TODO: perhaps add an option to clean up all but some existing ones.
# NOTE: affects the state directly of the model,
# TODO: add more options to control for file existence and what not.
# TODO: maybe add something to save many models. this can be useful.
# TODO: make it easy to get multiple models from it, and return multiple
# models to it.
# TODO: it is possible to add something to remove the whole folder in the end.
class Saver:

    def __init__(self, saver_folderpath):
        self.saver_folderpath = saver_folderpath
        self.name_to_cfg = {}
        self.name_to_save_fn = {}
        self.name_to_load_fn = {}

    def _get_filepath(self, name, use_json):
        filename = name + (".json" if use_json else '.pkl')
        filepath = tb_fs.join_paths([self.saver_folderpath, filename])
        return filepath

    def register(self, name, save_fn, load_fn, use_json=False):
        assert name not in self.name_to_cfg
        self.name_to_cfg[name] = {
            'save_fn': save_fn,
            'load_fn': load_fn,
            'use_json': use_json,
        }

    def save(self, name, x):
        cfg = self.name_to_cfg[name]
        out = cfg['save_fn'](x)
        filepath = self._get_filepath(name, cfg['use_json'])
        if cfg['use_json']:
            tb_io.write_jsonfile(out, filepath)
        else:
            tb_io.write_picklefile(out, filepath)

    def load(self, name, x):
        cfg = self.name_to_cfg[name]
        filepath = self._get_filepath(name, cfg['use_json'])
        if tb_fs.file_exists(filepath):
            if cfg['use_json']:
                out = tb_io.read_jsonfile(filepath)
            else:
                out = tb_io.read_picklefile(filepath)
            x = cfg['load_fn'](x, out)
        return x

    def clean(self, name):
        cfg = self.name_to_cfg[name]
        filepath = self._get_filepath(name, cfg['use_json'])
        tb_fs.delete_file(filepath, abort_if_notexists=False)

    def clean_all(self):
        for name in self.name_to_cfg:
            self.clean(name)

    # def unregister(self, name, delete_existing_checkpoint=False):
    #     self.name_to_save_fn.pop(name)
    #     self.name_to_load_fn.pop(name)
    #     filepath = self._get_filepath(name)
    #     if delete_existing_checkpoint
    #         tb_fs.delete_file(filepath, abort_if_notexists=False)


def get_best(eval_fns, minimize):
    best_i = None
    if minimize:
        best_v = np.inf
    else:
        best_v = -np.inf

    for i, fn in enumerate(eval_fns):
        v = fn()
        if minimize:
            if v < best_v:
                best_v = v
                best_i = i
        else:
            if v > best_v:
                best_v = v
                best_i = i

    return (best_i, best_v)


# useful for structuring training code.
def get_eval_fn(start_fn, train_fn, is_over_fn, end_fn):

    def eval_fn(d):
        start_fn(d)
        while not is_over_fn(d):
            train_fn(d)
        end_fn(d)

    return eval_fn