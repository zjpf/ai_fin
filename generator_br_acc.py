import csv
import pdb
import numpy as np


def to_float(s):
    try:
        return float(s)
    except ValueError:
        return 0.0

def gen_sample_v1(file_path, start_dt, end_dt, batch_size=4):
    f = open(file_path, "r")
    x, y = [], []
    f_csv = csv.reader(f)
    while True:
        while len(x) < batch_size:
            try:
                row = next(f_csv)
            except:
            #if row is None or len(row) == 0:
                f.seek(0)
                row = next(f_csv)
            label = row[-3]
            if label is not None and len(str(label)) > 0 and start_dt <= row[-1] <= end_dt:
                label = int(label)
            else:
                continue
            es = eval(row[1])
            ts = [list(map(to_float, e[2:])) for e in es if e[0] == 'br']
            x.append(ts)
            y.append(label)
        yield (np.array(x), np.array(y))
        x, y = [], []

def gen_sample(file_path, start_dt, end_dt, batch_size=4):
    f = open(file_path, "r")
    f_csv = csv.reader(f)
    for row in f_csv:
        #row = next(f_csv)
        label = row[-3]
        if label is not None and len(str(label)) > 0 and start_dt <= row[-1] <= end_dt:
            label = int(label)
        else:
            label = -1
            #continue
        es = eval(row[1])
        ts = [list(map(to_float, e[2:])) for e in es if e[0] == 'br']
        yield (np.array(ts), label, row[0] + ';' + row[-1])


def gen_sample__(file_path, start_dt, end_dt, batch_size=4):
    f = open(file_path, "r")
    f_csv = csv.reader(f)
    for row in f_csv:
        row = next(f_csv)
        label = row[-3]
        if label is not None and len(str(label)) > 0 and start_dt <= row[-1] <= end_dt:
            label = int(label)
        else:
            continue
        #es = eval(row[1])
        #ts = [list(map(to_float, e[2:])) for e in es if e[0] == 'br']
        yield row[1], label  # (np.array(ts), label)

def cvt_event(event, label):
    es = eval(event.numpy())
    ts = [list(map(to_float, e[2:])) for e in es if e[0] == 'br']
    return (np.array(ts), label)

def set_shape(x, y):
    x.set_shape([13, 264])
    y.set_shape([])
    return x, y

if __name__ == "__main__":
    import tensorflow as tf
    import functools
    fn, batch_size = 'b_2024q4_samples_v4', 4
    output_signature = (tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(), dtype=tf.float32))
    train_gen = functools.partial(gen_sample, file_path=fn, start_dt='2023-06-01', end_dt='2024-03-31', batch_size=batch_size)
    train_ds = tf.data.Dataset.from_generator(train_gen, output_signature=output_signature).map(lambda x, y: tf.py_function(cvt_event, (x,y), (tf.float32, tf.float32))).map(set_shape).prefetch(batch_size * 20).batch(batch_size)
    for tp in train_ds:
        pdb.set_trace()
        xy = tf.py_function(cvt_event, (tp[0][0], tp[1]), (tf.float32, tf.float32))
        pdb.set_trace()
    
    for x, y in gen_sample('b_2024q4_samples_v4_test', start_dt='2024-01-01', end_dt='2024-03-31'):
        pdb.set_trace()

