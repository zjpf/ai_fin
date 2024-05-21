# coding=utf-8
import tensorflow as tf
import random
import math
import csv
from datetime import datetime
import pickle
import time
import sys
import numpy as np
from sklearn.linear_model import LinearRegression
import pdb
csv.field_size_limit(1024 * 1024 * 1024)


class GenSample:
    def __init__(self, seq_length, start_dt, end_dt, version="v0_0"):
        self.version = version
        self.seq_length = seq_length
        self.start_dt = start_dt
        self.end_dt = end_dt
        self.data = []
        self.buffer_size = 200000
        self.event_types = ['p_v']
        self.event_type_attr_fn = {
            'p_v': [('max_price', 'dense'), ('min_price', 'dense'), ('opening_price', 'dense'),
                    ('closing_price', 'dense'), ('volume', 'dense'), ('amount', 'dense')]}
        self.attr_fns = set()
        self.seq_cate_fn = []
        self.seq_dense_fn = []
        for k in self.event_types:
            v = self.event_type_attr_fn[k]
            self.attr_fns.update(set(v))
            for fn_type in v:
                if fn_type[1] == "cate":
                    if fn_type[0] not in self.seq_cate_fn:
                        self.seq_cate_fn.append(fn_type[0])
                if fn_type[1] == "dense":
                    if fn_type[0] not in self.seq_dense_fn:
                        self.seq_dense_fn.append(fn_type[0])
        self.seq_cate_fn.extend(["mask", "event_dt"])
        self.sparse_fn = ["seq_length"]
        self.sparse_dense_fn = ['k_15d', 'max_15d', 'hk_15d', 'min_15d', 'min_5d', 'k_5d', 'hk_5d', 'max_5d']
        self.fn_id_map = {}
        self.id_map_path = 'id_map.pickle'

    @staticmethod
    def shuffle_write(f, records, mode):
        if len(records) <= 0:
            return
        random.shuffle(records)
        for line in records:
            f.write(line)
        print("write {} rn: {}".format(mode, len(records)))
        records.clear()

    def get_id_index(self, fn, k, mode):
        if mode == 'train':
            od = self.fn_id_map.setdefault(fn, {})
            if fn == "event_type":
                dn = len(od) + 1  # 0留给cls标记
            else:
                dn = len(od)
            ov = od.setdefault(k, dn)
        else:
            od = self.fn_id_map.get(fn, {})
            ov = od.get(k, -1)
        return ov

    def get_fn_voc(self):
        fn_voc = {}
        for fn, id_map in self.fn_id_map.items():
            if fn == "event_type":
                fn_voc[fn] = len(id_map) + 1
            else:
                fn_voc[fn] = len(id_map)
        print(fn_voc)

    def save_id_map(self):
        print(self.fn_id_map)
        with open(self.id_map_path, 'wb') as f:
            pickle.dump(self.fn_id_map, f, pickle.HIGHEST_PROTOCOL)

    def load_id_map(self):
        with open(self.id_map_path, 'rb') as f:
            self.fn_id_map = pickle.load(f)

    @staticmethod
    def get_label(events, ei):
        cur_price = float(events[ei][4])
        arr_5d = np.array([float(events[i][4]) / cur_price for i in range(ei, min(ei + 2, len(events)))])
        arr_15d = np.array([float(events[i][4]) / cur_price for i in range(ei, min(ei + 4, len(events)))])
        m_5d = [999] * 2 if len(arr_5d) == 0 else [arr_5d.max() - 1.0, arr_5d.min() - 1.0]
        m_15d = [999] * 2 if len(arr_15d) == 0 else [arr_15d.max() - 1.0, arr_15d.min() - 1.0]
        # 线性回归斜率为label
        x5, x15 = np.expand_dims(np.arange(len(arr_5d)), -1), np.expand_dims(np.arange(len(arr_15d)), -1)
        if len(arr_5d) >= 2:
            reg_5, reg_15 = LinearRegression().fit(x5, arr_5d), LinearRegression().fit(x15, arr_15d)
            k = [reg_5.coef_[0], reg_15.coef_[0]]
        else:
            k = [999]*2
        hist_5d = np.array([float(events[i][4]) / cur_price for i in range(max(0, ei-10), ei)])
        hist_15d = np.array([float(events[i][4]) / cur_price for i in range(max(0, ei-20), ei)])
        hx5, hx15 = np.expand_dims(np.arange(len(hist_5d)), -1), np.expand_dims(np.arange(len(hist_15d)), -1)
        if len(hist_5d) >= 10:
            reg_5, reg_15 = LinearRegression().fit(hx5, hist_5d), LinearRegression().fit(hx15, hist_15d)
            hk = [reg_5.coef_[0], reg_15.coef_[0]]
        else:
            hk = [999]*2
        return m_5d + m_15d + k + hk

    def get_xy(self, raw_file, mode='train'):
        data_tf = "./{}_data_{}.tf".format(mode, self.version)
        tf_writer = tf.io.TFRecordWriter(data_tf)
        if mode != 'train':
            self.load_id_map()
        # 构建特征向量
        with open(raw_file, encoding="utf-8") as f:
            for row in csv.reader(f):
                fn_lv = {fn: [] for fn, f_type in list(self.attr_fns)}
                events = eval(row[1])
                for ei, tp in enumerate(events):
                    event_dt = tp[0]
                    for i, fn_type in enumerate(self.event_type_attr_fn['p_v']):
                        if fn_type[1] == "dense":
                            if tp[i + 1] not in ["null", ""]:
                                fv = float(tp[i + 1])
                                fn_lv[fn_type[0]].append(fv)
                            else:
                                raise ValueError
                    # 生成一条样本
                    if self.start_dt <= event_dt <= self.end_dt and ei >= 40 and ei+1<len(events):
                        max_5d, min_5d, max_15d, min_15d, k_5d, k_15d, hk_5d, hk_15d = GenSample.get_label(events, ei+1)
                        lbs = {"max_5d": max_5d, "min_5d": min_5d, "max_15d": max_15d, "min_15d": min_15d,
                               "k_5d": k_5d, "k_15d": k_15d, "hk_5d": hk_5d, "hk_15d": hk_15d}
                        padded_fn_lv = {}
                        for fn, lv in fn_lv.items():
                            f_len = len(lv)
                            rev_lv = lv[-self.seq_length:]
                            rev_lv.reverse()
                            padded_fn_lv[fn] = rev_lv + [0] * (self.seq_length - len(lv))
                            p_len = min(f_len, self.seq_length)
                            # c = padded_fn_lv[fn][0]
                            for pi in range(0, p_len):  # 相对最近一天的波动比例
                                # padded_fn_lv[fn][pi] = math.log10(padded_fn_lv[fn][pi] / c + 1.0)  # p_len-1
                                p_v = padded_fn_lv[fn][min(pi + 1, p_len - 1)]
                                padded_fn_lv[fn][pi] = (padded_fn_lv[fn][pi] - p_v) / p_v
                                # padded_fn_lv[fn][pi] = math.log10((padded_fn_lv[fn][pi]-p_v)/p_v + 1.0)
                        for fn, lv in lbs.items():
                            padded_fn_lv[fn] = [lv]
                        padded_fn_lv["seq_length"] = [p_len]
                        padded_fn_lv["mask"] = [1] + [1] * (p_len - 1) + [0] * (self.seq_length - p_len)
                        padded_fn_lv["event_dt"] = [0] + list(range(1, p_len)) + [0] * (self.seq_length - p_len)
                        # GenSample.check(fn_lv)
                        value_dict = {}
                        for name in self.sparse_fn:
                            value_dict[name] = tf.train.Feature(
                                int64_list=tf.train.Int64List(value=padded_fn_lv.get(name)))
                        for name in self.sparse_dense_fn:
                            value_dict[name] = tf.train.Feature(
                                float_list=tf.train.FloatList(value=padded_fn_lv.get(name)))
                        for name in self.seq_cate_fn:
                            value_dict[name] = tf.train.Feature(
                                int64_list=tf.train.Int64List(value=padded_fn_lv.get(name)))
                        for name in self.seq_dense_fn:
                            value_dict[name] = tf.train.Feature(
                                float_list=tf.train.FloatList(value=padded_fn_lv.get(name)))
                        if mode == 'oot':
                            info = (','.join(
                                [row[0], event_dt, '%.3f' % max_5d, '%.3f' % max_15d, '%.3f' % min_5d, '%.3f' % min_15d,
                                 '%.3f' % hk_5d, '%.3f' % hk_15d, '%.3f' % k_5d, '%.3f' % k_15d, ",".join(map(str, tp[1:]))])).encode(encoding='utf-8')
                        else:
                            info = (','.join([row[0], event_dt])).encode(encoding='utf-8')
                        value_dict['info'] = tf.train.Feature(bytes_list=tf.train.BytesList(value=[info]))
                        tf_example = tf.train.Example(features=tf.train.Features(feature=value_dict))
                        tf_serialized = tf_example.SerializeToString()
                        self.data.append(tf_serialized)
                        #pdb.set_trace()
                        if len(self.data) >= self.buffer_size:
                            GenSample.shuffle_write(tf_writer, self.data, mode)
        GenSample.shuffle_write(tf_writer, self.data, mode)
        tf_writer.close()
        self.get_fn_voc()
        if mode == 'train':
            self.save_id_map()
        print(self.seq_cate_fn, self.seq_dense_fn)


if __name__ == "__main__":
    if len(sys.argv) >= 6:
        train_st, train_et, oot_st, oot_et, ver = sys.argv[1:6]
    else:
        train_st, train_et, oot_st, oot_et, ver = '20140101', '20240315', '20240316', '20240413', 'v1_2'
    g = GenSample(seq_length=700, start_dt=train_st, end_dt=train_et, version=ver)
    g.get_xy(raw_file="2024_04_13.csv", mode='train')
    g1 = GenSample(seq_length=700, start_dt=oot_st, end_dt=oot_et, version=ver)
    g1.get_xy(raw_file="2024_04_13.csv", mode='oot')

