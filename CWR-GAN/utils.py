from __future__ import division
import tensorflow as tf, numpy as np

def modular_slice(l, start, end, shuffle_at_end=True):
    if l == []: return []
    elif end - start >= len(l): return l
    s, e = start % len(l), end % len(l)
    if s < e: return l[s:e]
    res = l[s:] + l[:e]
    if shuffle_at_end: np.random.shuffle(l)
    return res

def add_summaries(writer, tags, values, step):
    for i in xrange(len(tags)):
        writer.add_summary(tf.Summary(
            value=[tf.Summary.Value(tag=tags[i], simple_value=values[i])]), step)

def intersect(values_d, keys):
    out_keys = set(values_d.keys()) & keys
    out_d = {}
    for key in out_keys:
      out_d[key] = values_d[key]
    return out_d
def div_dict(running_d, scalar): return {key: value/scalar for key, value in running_d.items()}
def sum_dict(running_d, new_d):
    for key, value in new_d.items(): running_d[key] += value
