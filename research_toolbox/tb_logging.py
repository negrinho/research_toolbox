### date and time
import datetime
import sys
import psutil
import pprint
import platform
import time
import os
import research_toolbox.tb_utils as tb_ut
import research_toolbox.tb_resources as tb_rs


def convert_between_time_units(x, src_units='seconds', dst_units='hours'):
    d = {}
    d['seconds'] = 1.0
    d['minutes'] = 60.0 * d['seconds']
    d['hours'] = 60.0 * d['minutes']
    d['days'] = 24.0 * d['hours']
    d['weeks'] = 7.0 * d['days']
    d['miliseconds'] = d['seconds'] * 1e-3
    d['microseconds'] = d['seconds'] * 1e-6
    d['nanoseconds'] = d['seconds'] * 1e-9
    return (x * d[src_units]) / d[dst_units]


def memory_process(pid, units='megabytes'):
    psutil_p = psutil.Process(pid)
    mem_p = psutil_p.memory_info()[0]
    return tb_rs.convert_between_byte_units(mem_p, dst_units=units)


# TODO: do not just return a string representation, return the numbers.
def now(omit_date=False, omit_time=False, time_before_date=False):
    assert (not omit_time) or (not omit_date)

    d = datetime.datetime.now()
    date_s = ''
    if not omit_date:
        date_s = "%d-%.2d-%.2d" % (d.year, d.month, d.day)
    time_s = ''
    if not omit_time:
        time_s = "%.2d:%.2d:%.2d" % (d.hour, d.minute, d.second)

    vs = []
    if not omit_date:
        vs.append(date_s)
    if not omit_time:
        vs.append(time_s)

    # creating the string
    if len(vs) == 2 and time_before_date:
        vs = vs[::-1]
    s = '|'.join(vs)

    return s


def now_dict():
    x = datetime.datetime.now()
    return tb_ut.create_dict(
        ['year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond'],
        [x.year, x.month, x.day, x.hour, x.minute, x.second, x.microsecond])


### logging
class TimeTracker:

    def __init__(self):
        self.init_time = time.time()
        self.last_registered = self.init_time

    def time_since_start(self, units='seconds'):
        now = time.time()
        elapsed = now - self.init_time

        return convert_between_time_units(elapsed, dst_units=units)

    def time_since_last(self, units='seconds'):
        now = time.time()
        elapsed = now - self.last_registered
        self.last_registered = now

        return convert_between_time_units(elapsed, dst_units=units)


class TimerManager:

    def __init__(self):
        self.init_time = time.time()
        self.last_registered = self.init_time
        self.name_to_timer = {}

    def create_timer(self, timer_name, abort_if_timer_exists=True):
        assert not abort_if_timer_exists or timer_name not in self.name_to_timer
        start_time = time.time()
        self.name_to_timer[timer_name] = {
            'start': start_time,
            'tick': start_time
        }

    def create_timer_event(self,
                           timer_name,
                           event_name,
                           abort_if_event_exists=True):
        assert not timer_name == 'tick'
        timer = self.name_to_timer[timer_name]
        assert not abort_if_event_exists or event_name not in timer
        timer[event_name] = time.time()

    def tick_timer(self, timer_name):
        self.name_to_timer[timer_name]['tick'] = time.time()

    def get_time_since_event(self, timer_name, event_name, units='seconds'):
        delta = time.time() - self.name_to_timer[timer_name][event_name]
        return convert_between_time_units(delta, dst_units=units)

    def get_time_between_events(self,
                                timer_name,
                                earlier_event_name,
                                later_event_name,
                                units='seconds'):
        timer = self.name_to_timer[timer_name]
        delta = timer[earlier_event_name] - timer[later_event_name]
        assert delta >= 0.0
        return convert_between_time_units(delta, dst_units=units)

    def get_time_since_last_tick(self, timer_name, units='seconds'):
        delta = time.time() - self.name_to_timer[timer_name]['tick']
        return convert_between_time_units(delta, dst_units=units)


class MemoryTracker:

    def __init__(self):
        self.last_registered = 0.0
        self.max_registered = 0.0

    def memory_total(self, units='megabytes'):
        mem_now = memory_process(os.getpid(), units)
        if self.max_registered < mem_now:
            self.max_registered = mem_now

        return tb_rs.convert_between_byte_units(mem_now, dst_units=units)

    def memory_since_last(self, units='megabytes'):
        mem_now = self.memory_total('bytes')

        mem_dif = mem_now - self.last_registered
        self.last_registered = mem_now

        return tb_rs.convert_between_byte_units(mem_dif, dst_units=units)

    def memory_max(self, units='megabytes'):
        return tb_rs.convert_between_byte_units(
            self.max_registered, dst_units=units)


def print_time(timer, prefix_str='', units='seconds'):
    print(('%s%0.2f %s since start.' %
          (prefix_str, timer.time_since_start(units=units), units)))
    print(("%s%0.2f %s seconds since last call." %
          (prefix_str, timer.time_since_last(units=units), units)))


def print_memory(memer, prefix_str='', units='megabytes'):
    print(('%s%0.2f %s total.' % (prefix_str, memer.memory_total(units=units),
                                 units.upper())))
    print(("%s%0.2f %s since last call." %
          (prefix_str, memer.memory_since_last(units=units), units.upper())))


def print_memorytime(memer,
                     timer,
                     prefix_str='',
                     mem_units='megabytes',
                     time_units='seconds'):
    print_memory(memer, prefix_str, units=mem_units)
    print_time(timer, prefix_str, units=time_units)


def print_oneliner_memorytime(memer,
                              timer,
                              prefix_str='',
                              mem_units='megabytes',
                              time_units='seconds'):

    print(('%s (%0.2f %s last; %0.2f %s total; %0.2f %s last; %0.2f %s total)' %
          (prefix_str, timer.time_since_last(units=time_units), time_units,
           timer.time_since_start(units=time_units), time_units,
           memer.memory_since_last(units=mem_units), mem_units.upper(),
           memer.memory_total(units=mem_units), mem_units.upper())))


class Logger:

    def __init__(self, filepath, append_to_file=False,
                 capture_all_output=False):
        mode = 'a' if append_to_file else 'w'
        self.f = open(filepath, mode)
        if capture_all_output:
            capture_output(self.f)

    def log(self, s, description=None, preappend_datetime=False):
        if preappend_datetime:
            self.f.write(now() + '\n')

        if description is not None:
            self.f.write(description + '\n')

        if not isinstance(s, str):
            s = pprint.pformat(s)

        self.f.write(s + '\n')


# check if files are flushed automatically upon termination, or there is
# something else that needs to be done. or maybe have a flush flag or something.
def capture_output(f, capture_stdout=True, capture_stderr=True):
    """Takes a file as argument. The file is managed externally."""
    if capture_stdout:
        sys.stdout = f
    if capture_stderr:
        sys.stderr = f


def node_information():
    return platform.node()