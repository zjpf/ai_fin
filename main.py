import sys
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from gen_xy_k import GenSample
from model_fin import HeteroTransformer
import csv
import pdb

batch_size, seq_len = 128*3, 700
g = GenSample(seq_length=seq_len, start_dt='19900101', end_dt='20220901')
fn_voc_size = eval("{}")
a1, a2, a3, a4 = 18, 4, 3, 6
label = None
thre = {"max_5d": [-0.01, 0.15], "min_5d": [-0.03, 0.03], "med_5d,max_20d": [-0.01, 0.2], "min_20d":[-0.05, 0.03]}


def get_feature_parser():
    parse_dict = {}
    for name in g.sparse_fn:
        parse_dict[name] = tf.io.FixedLenFeature(shape=(), dtype=tf.int64)
    for name in g.sparse_dense_fn:
        parse_dict[name] = tf.io.FixedLenFeature(shape=(), dtype=tf.float32)
    for name in g.seq_cate_fn:
        parse_dict[name] = tf.io.FixedLenFeature(shape=(seq_len,), dtype=tf.int64)
    for name in g.seq_dense_fn:
        parse_dict[name] = tf.io.FixedLenFeature(shape=(seq_len,), dtype=tf.float32)
    parse_dict['info'] = tf.io.FixedLenFeature(shape=(), dtype=tf.string)
    return parse_dict


def parse_feature_label(example_proto):
    parse_dict = get_feature_parser()
    parsed_example = tf.io.parse_single_example(example_proto, parse_dict)
    y = parsed_example[label]
    #yi = tf.cast(y*100, tf.int32)
    #y = tf.where(yi<2, 0, 1)*tf.where(yi<99000, 1, yi)
    #y = tf.where(y<-0.02, 1, 0)+tf.where(y>0.02, 2, 0)+tf.where(y>99, 99, 0)
    #y= tf.where(y>-0.02, 1,0)+tf.where(y>-0.01,1,0)+tf.where(y>0.01,1,0)+tf.where(y>0.02,1,0)+tf.where(y>99, 99, 0)
    y= tf.where(y>-0.02, [1,0,0,0], [0,0,0,0])+tf.where(y>-0.01,[0,1,0,0],[0,0,0,0])+ \
      tf.where(y>0.01,[0,0,1,0], [0,0,0,0])+tf.where(y>0.02,[0,0,0,1],[0,0,0,0])+tf.where(y>99, [99,99,99,99], [0,0,0,0])
    del parsed_example[label]
    del parsed_example['info']
    out = {}
    for k, v in parsed_example.items():
        if k in ['seq_length', 'event_dt', 'mask']:
            out[k]=v
        else:
            out[k]=tf.cast(tf.cast(v*100, tf.int32), tf.float32)*0.01
        #out[k]=tf.clip_by_value(tf.cast(tf.cast(v*100, tf.int32), tf.float32)*0.01, -1.0, 3.5)
    return out, y


def parse_feature_label_info(example_proto):
    parse_dict = get_feature_parser()
    parsed_example = tf.io.parse_single_example(example_proto, parse_dict)
    y = parsed_example[label]
    #yi = tf.cast(y*100, tf.int32)
    #y = tf.where(yi<2, 0, 1)*tf.where(yi<99000, 1, yi)
    #y = tf.where(y<-0.02, 1, 0)+tf.where(y>0.02, 2, 0)+tf.where(y>99, 99, 0)
    y= tf.where(y>-0.02, [1,0,0,0], [0,0,0,0])+tf.where(y>-0.01,[0,1,0,0],[0,0,0,0])+\
      tf.where(y>0.01,[0,0,1,0], [0,0,0,0])+tf.where(y>0.02,[0,0,0,1],[0,0,0,0])+tf.where(y>99, [99,99,99,99], [0,0,0,0])
    info = parsed_example['info']
    del parsed_example[label]
    del parsed_example['info']
    out = {}
    for k, v in parsed_example.items():
        if k in ['seq_length', 'event_dt', 'mask']:
            out[k]=v
        else:
            out[k]=tf.cast(tf.cast(v*100, tf.int32), tf.float32)*0.01
    return (out, y, info)


def get_data_set(file_name, num_replica=1, with_info=False):
    ds_raw = tf.data.TFRecordDataset([file_name])
    if with_info:
        ds = ds_raw.map(parse_feature_label_info, num_parallel_calls=num_replica)
        #return ds.filter(lambda x,y,z: tf.math.less(y, 6)).prefetch(batch_size * num_replica).batch(batch_size)
        return ds.filter(lambda x,y,z: tf.math.less(y[0], 6)).prefetch(batch_size * num_replica).batch(batch_size)
    else:
        ds = ds_raw.map(parse_feature_label, num_parallel_calls=num_replica)
        return ds.filter(lambda x,y: tf.math.less(y[0], 6)).prefetch(batch_size * num_replica).batch(batch_size)


class ModelSaver(tf.keras.callbacks.Callback):
    def __init__(self, monitor_metric, seq_model, model_path='saved_model', mode='min'):
        self.monitor_metric = monitor_metric
        if mode == 'min':
            self.metric_value = sys.maxsize
        elif mode == 'max':
            self.metric_value = -sys.maxsize
        self.seq_model = seq_model
        self.mode = mode
        self.model_path = model_path

    def on_epoch_end(self, epoch, logs={}):
        if ((self.mode == 'min' and logs[self.monitor_metric] <= self.metric_value) or
                (self.mode == 'max' and logs[self.monitor_metric] >= self.metric_value)):
            self.seq_model.save_weights(self.model_path)
            self.metric_value = logs[self.monitor_metric]
            print("Epoch: {}, Metric_value: {}".format(epoch, self.metric_value))


def auc(y_true, y_prob):
    return tf.py_function(roc_auc_score_fixed, (y_true, y_prob), tf.double)


def roc_auc_score_fixed(y_true, y_prob):
    if len(np.unique(y_true)) == 1:
        print("Warn: only one class in y_true")
        return accuracy_score(y_true, np.rint(y_prob))
    return roc_auc_score(y_true, y_prob)


def train(train_data_tf, test_data_tf, version):
    train_ds = get_data_set(train_data_tf)
    test_ds = get_data_set(test_data_tf)
    model = HeteroTransformer(seq_length=seq_len, fn_voc_size=fn_voc_size, dnn_dim=a1, num_hidden_layers=a2, num_attention_heads=a3, size_per_head=a4).model
    callbacks = [EarlyStopping(monitor='val_auc', patience=3, mode="max"),
                 ReduceLROnPlateau(monitor='val_auc', factor=0.3, patience=1, min_lr=1e-8, mode='max'),
                 ModelSaver("val_auc", model, model_path="saved_model/{}.h5".format(version), mode='max')]
    callbacks = [EarlyStopping(monitor='val_loss', patience=3, mode="min"),
                 ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=1, min_lr=1e-8, mode='min'),
                 ModelSaver("val_loss", model, model_path="saved_model/{}.weights.h5".format(version), mode='min')]
    try:
        os.mkdir('saved_model')
    except:
        print("saved_model exists")
    model.compile("adam", "binary_crossentropy", metrics=['auc', 'binary_crossentropy'])
    #model.compile("adam", "sparse_categorical_crossentropy", metrics=['sparse_categorical_crossentropy'])
    history = model.fit(train_ds, epochs=100, verbose=1, validation_data=test_ds, callbacks=callbacks)


def predict(data_tf, v):
    model = HeteroTransformer(seq_length=seq_len, fn_voc_size=fn_voc_size, dnn_dim=a1, num_hidden_layers=a2,
                              num_attention_heads=a3, size_per_head=a4).model
    model.load_weights("saved_model/{}.weights.h5".format(v))
    oot_ds = get_data_set(data_tf, with_info=True)
    with open('score_oot_{}'.format(v), 'w') as f:
        for x, y, info in oot_ds:
            y_p = model.predict(x)
            #pdb.set_trace()
            for bi in range(len(info)):
                f.write("{}".format(info[bi].numpy().decode('utf-8'))[2:-1] + ",{},{}\n".format(y[bi], ','.join(map(str, y_p[bi])) ))


if __name__ == "__main__":
    if not True:
        label = 'k_15d'
        oot_ds = get_data_set("oot_data_v0_s.tf", with_info=True)
        #ys = 0
        y1, d_cnt = 0, {}
        #pdb.set_trace()
        for x, y, info in oot_ds:
        #    ys += 1
            pdb.set_trace()
            for v in y:
                ov = d_cnt.setdefault(int(v*100), 0)
                d_cnt[int(v*100)] = ov+1
        pdb.set_trace()
        #for bi in range(len(info)):
        #    y1 += y[bi]
    #print(sorted(d_cnt.items(), key=lambda x: x[0]))
    #pdb.set_trace()
    if len(sys.argv) >= 3:
        label, ver, lb_ver, gpu = sys.argv[1:5]
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    train(train_data_tf='train_data_{}.tf'.format(ver), test_data_tf='oot_data_{}.tf'.format(ver), version=lb_ver)
    predict(data_tf='oot_data_{}.tf'.format(ver), v=lb_ver)
    print("End")

