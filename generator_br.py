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
        row = next(f_csv)
        label = row[-3]
        if label is not None and len(str(label)) > 0 and start_dt <= row[-1] <= end_dt:
            label = int(label)
        else:
            continue
        es = eval(row[1])
        ts = [list(map(to_float, e[2:])) for e in es if e[0] == 'br']
        yield (np.array(ts), label)


def gen_sample_(file_path, start_dt, end_dt, batch_size=4):
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
        yield row[1]  # (np.array(ts), label)

def cvt_event(event):
    es = eval(event)
    ts = [list(map(to_float, e[2:])) for e in es if e[0] == 'br']

if __name__ == "__main__":
    for x, y in gen_sample('b_2024q4_samples_v4_test', start_dt='2024-01-01', end_dt='2024-03-31'):
        pdb.set_trace()

