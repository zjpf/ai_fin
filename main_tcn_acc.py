import sys
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score, accuracy_score
import numpy as np
from gen_xy_br import GenXy
from model_ms import HeteroTransformer
import pdb
batch_size, ver = 128*5, 'b_2024q4_br'
g = GenXy(start_dt=None, end_dt=None, version=ver)
fn_voc_size = g.get_fn_voc()
from tcn import TCN
import numpy as np
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense
import tensorflow.keras as keras
TruncatedNormal = keras.initializers.TruncatedNormal


def get_feature_parser():
    parse_dict = {}
    for name in g.sparse_fn:
        parse_dict[name] = tf.io.FixedLenFeature(shape=(), dtype=tf.int64)
    for name, sn in g.seq_cate_fn.items():
        parse_dict[name] = tf.io.FixedLenFeature(shape=(sn,), dtype=tf.int64)
    for name, sn in g.seq_dense_fn.items():
        parse_dict[name] = tf.io.FixedLenFeature(shape=(sn,), dtype=tf.float32)
    for name, sn in g.seq_words_fn.items():
        parse_dict[name] = tf.io.FixedLenFeature(shape=(sn*10*100,), dtype=tf.float32)
    for name, sn in g.seq_vec_fn.items():
        parse_dict[name] = tf.io.FixedLenFeature(shape=(sn * 24,), dtype=tf.float32)
    parse_dict['info'] = tf.io.FixedLenFeature(shape=(), dtype=tf.string)
    return parse_dict


def parse_feature_label(example_proto):
    parse_dict = get_feature_parser()
    parsed_example = tf.io.parse_single_example(example_proto, parse_dict)
    y = parsed_example['label']
    # del parsed_example['label']
    del parsed_example['info']
    del parsed_example['label']
    #a = parsed_example['event_dt_app_ins']
    #parsed_example['event_dt_app_ins'] = tf.clip_by_value( a // 30, 0, 400)
    #parsed_example['event_dt_app_ins'] = tf.clip_by_value(tf.where(a<=731, a, 731 + (a - 731) // 30), 0, 831)
    #out = {}
    #for k, v in parsed_example.items():
    #    if k in ['sx_duration', 'sx_limit', 'overdue_amount', 'balance', 'repayment_amount', 'bill_amount', 'loan_amount', 'credit_limit', 'available_amt', 'irr_rate', 'loan_amt']:
    #        out[k] = v+tf.random.uniform(v.shape, minval=-0.041, maxval=0.041)
    #    else:
    #        out[k] = v
    return parsed_example, y


def parse_feature_label_info(example_proto):
    parse_dict = get_feature_parser()
    parsed_example = tf.io.parse_single_example(example_proto, parse_dict)
    y = parsed_example['label']
    info = parsed_example['info']
    # del parsed_example['label']
    #a = parsed_example['event_dt_app_ins']
    #parsed_example['event_dt_app_ins'] = tf.clip_by_value( a // 30, 0, 400)
    #parsed_example['event_dt_app_ins'] = tf.clip_by_value(tf.where(a<=731, a, 731 + (a - 731) // 30), 0, 831)
    del parsed_example['info']
    del parsed_example['label']
    return parsed_example, y, info


def get_data_set(file_name, num_replica=30, with_info=False, repeat=False, pos_sr=None, neg_sr=None):
    ds_raw = tf.data.TFRecordDataset([file_name])
    if with_info:
        ds = ds_raw.map(parse_feature_label_info, num_parallel_calls=num_replica)
    else:
        ds = ds_raw.map(parse_feature_label, num_parallel_calls=num_replica)
    if repeat:
        return ds.prefetch(batch_size * num_replica).filter(lambda x,y: tf.less(tf.random.uniform(y.shape)*(1-tf.cast(y, tf.float32)), 0.3)).batch(batch_size).repeat()
    else:
        if with_info:
            return ds.prefetch(batch_size * num_replica).batch(batch_size)
        if pos_sr is None and  neg_sr is None:
            return ds.filter(lambda x,y: y>=0).prefetch(batch_size * num_replica).batch(batch_size)
        else:
            pos_sr = pos_sr if pos_sr is not None else 1
            neg_sr = neg_sr if neg_sr is not None else 1
            return ds.filter(lambda x,y: tf.logical_and(tf.less(tf.random.uniform(y.shape)*tf.cast(y, tf.float32), pos_sr), tf.less(tf.random.uniform(y.shape)*(1-tf.cast(y, tf.float32)), neg_sr))).prefetch(batch_size * num_replica).batch(batch_size)
        #return ds.prefetch(batch_size * num_replica).batch(batch_size)


class ModelSaver(tf.keras.callbacks.Callback):
    def __init__(self, monitor_metric, seq_model, model_path='saved_model', mode='min', i=None):
        self.monitor_metric = monitor_metric
        if mode == 'min':
            self.metric_value = sys.maxsize
        elif mode == 'max':
            self.metric_value = -sys.maxsize
        self.seq_model = seq_model
        self.mode = mode
        self.model_path = model_path
        self.i = i

    def on_epoch_end(self, epoch, logs={}):
        if ((self.mode == 'min' and logs[self.monitor_metric] <= self.metric_value) or
                (self.mode == 'max' and logs[self.monitor_metric] >= self.metric_value)):
            if epoch >= 1:
                os.rename('saved_model/model_br.h5',
                          'saved_model/model_{}_epoch_{}.h5'.format(self.i, epoch - 1))
            self.seq_model.save('saved_model/model_br.h5', overwrite=True)
            self.metric_value = logs[self.monitor_metric]
            print("Epoch: {}, Metric_value: {}".format(epoch, self.metric_value))


def auc(rn=None):
    #@tf.autograph.experimental.do_not_convert
    def auc_in(y_true, y_prob):
        #pdb.set_trace()
        if rn is not None:
            y_prob = y_prob[rn, :]
        a = tf.py_function(roc_auc_score_fixed, (y_true, y_prob), tf.double)
        #a.set_shape((batch_size,))
        return a

    def roc_auc_score_fixed(y_true, y_prob):
        if len(np.unique(y_true)) == 1:
            print("Warn: only one class in y_true")
            return accuracy_score(y_true, np.rint(y_prob))
        return roc_auc_score(y_true, y_prob)
    return auc_in


def train( version='v0_0', pos_sr=None, neg_sr=None):
    from generator_br_acc import gen_sample, cvt_event, set_shape
    import functools
    fn = 'b_2024q4_samples_v4'
    output_signature = (tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(), dtype=tf.float32))
    train_gen = functools.partial(gen_sample, file_path=fn, start_dt='2023-06-01', end_dt='2024-03-31', batch_size=batch_size)
    test_gen = functools.partial(gen_sample, file_path=fn, start_dt='2024-04-01', end_dt='2024-05-31', batch_size=batch_size)
    train_ds = tf.data.Dataset.from_generator(train_gen, output_signature=output_signature).map(lambda x, y: tf.py_function(cvt_event, (x,y), (tf.float32, tf.float32)), num_parallel_calls=tf.data.AUTOTUNE).map(set_shape, num_parallel_calls=tf.data.AUTOTUNE).prefetch(batch_size * 20).batch(batch_size)
    test_ds = tf.data.Dataset.from_generator(test_gen, output_signature=output_signature).map(lambda x, y: tf.py_function(cvt_event, (x,y), (tf.float32, tf.float32)), num_parallel_calls=tf.data.AUTOTUNE).map(set_shape, num_parallel_calls=tf.data.AUTOTUNE).prefetch(batch_size * 20).batch(batch_size)
    #a = next(test_ds.make_one_shot_iterator())
    #pdb.set_trace()
    #for tp in train_ds:
    #    print(tp)
    #    break
    #    pdb.set_trace()
    #print([inp.name.split(':')[0] for inp in model.inputs])
    #return
    #strategy = tf.distribute.MirroredStrategy()
    #with strategy.scope():
    #model = HeteroTransformer(sn_tp=g.sn_tp, fn_voc_size=fn_voc_size, dnn_dim=32, num_hidden_layers=4, num_attention_heads=4, size_per_head=8, fns_tp=g.fns_tp).model

    inp = Input(shape=(13, 264), name='input_ts', dtype='float32')
    tcn_out = TCN(nb_filters=64, return_sequences=False, padding='causal')(inp)
    dnn_out = Dense(64, activation='relu', kernel_initializer=TruncatedNormal(0.02))(tcn_out)
    predict_out = Dense(1, activation='sigmoid', kernel_initializer=TruncatedNormal(0.02))(dnn_out)
    model = Model(inputs=[inp], outputs=[predict_out])
    callbacks = [EarlyStopping(monitor='val_auc', patience=3, mode="max"),
                                 ReduceLROnPlateau(monitor='val_auc', factor=0.3, patience=1, min_lr=1e-8, mode='max'),
                                                  ModelSaver("val_auc", model, model_path="saved_model/", mode='max', i=version)]
    model.compile(optimizer=tf.keras.optimizers.Adam(3e-4), loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC()])
    #import pickle
    #with open("../pre_train/data/saved_model/id_embedding_5.8665_0.3063.pickle", 'rb') as f:
    #    id_emb = pickle.load(f)
    #pdb.set_trace()
    #model.layers[3].set_weights([id_emb[:-2]])
    #model.layers[3].trainable = False
    #pdb.set_trace()
    model.load_weights('saved_model/model_br_0.61426.h5')
    model.fit(train_ds, epochs=50, verbose=1, validation_data=test_ds, callbacks=callbacks) #, steps_per_epoch=10, validation_steps=10)
    os.rename('saved_model/model_br.h5', 'saved_model/model_{}.h5'.format(version))
    print('End')


def predict( out_file, version):
    inp = Input(shape=(13, 264), name='input_ts', dtype='float32')
    tcn_out = TCN(nb_filters=64, return_sequences=False, padding='causal')(inp)
    dnn_out = Dense(64, activation='relu', kernel_initializer=TruncatedNormal(0.02))(tcn_out)
    predict_out = Dense(1, activation='sigmoid', kernel_initializer=TruncatedNormal(0.02))(dnn_out)
    model = Model(inputs=[inp], outputs=[predict_out])
    model.load_weights('saved_model/model_br_0.61426.h5')
    #pdb.set_trace()
    from generator_br_acc import gen_sample
    import functools
    fn = 'b_2024q4_samples_v4'
    output_signature = (tf.TensorSpec(shape=(13, 264), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.float32), tf.TensorSpec(shape=(), dtype=tf.string))
    train_gen = functools.partial(gen_sample, file_path=fn, start_dt='2023-06-01', end_dt='2024-08-31', batch_size=batch_size)
    oot_ds = tf.data.Dataset.from_generator(train_gen, output_signature=output_signature).prefetch(batch_size * 20).batch(batch_size)
    #oot_ds = get_data_set(data_tf, with_info=True)
    with open(out_file, 'w') as f:
        for x, y, info in oot_ds:
            y_p = model.predict(x)
            #pdb.set_trace()
            for bi in range(len(info)):
                #emb = ",".join(map(str, y_p[1][bi][0]))
                f.write("{}".format(info[bi])[2:-1] + ";{};{:.7f}\n".format(y[bi], y_p[bi][0]))


if __name__ == "__main__":
    import os
    import sys
    if len(sys.argv) > 1:  # 带参数为提取样本+训练: 原始样本文件、训练起止日期、oot起止日期、模型版本标识、使用gpus号
        ver, gpus, mode = sys.argv[1:4]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus
        #tf.executing_eagerly, ver = True, 'b_sf'
        #predict(data_tf='train_data_{}.tf'.format(ver), out_file='score_oot_{}'.format(ver), version=ver)
        if mode.find('train') >= 0:
            #for neg_sr, pos_sr in [(0.05, 0.2), (0.05, 0.5), (0.1, 0.2), (0.1, 0.5), (0.1, 0.8), (0.2, 0.5)]:
            train(version=ver)
        if mode.find('predict') >= 0:
            predict( out_file='score_train_{}_tcn_0.614_quan'.format(ver), version='b_2024q4_br')
            #predict(data_tf='oot_data_{}.tf'.format(ver), out_file='score_oot_{}_v1_br_ep11'.format(ver), version='b_2024q4_br')
            #predict(data_tf='oot_data_{}.tf_202406_08'.format(ver), out_file='score_oot_{}_v2_zx_ep11_202406_08'.format(ver), version='b_2024q4')
            #predict(data_tf='oot_{}.tf'.format(ver), out_file='score_oot_{}'.format(ver), version='gjz_data_b_2024q1_m4')
            #predict(data_tf='train_{}.tf_all'.format(ver), out_file='score_train_{}'.format(ver), version=ver)
            #predict(data_tf='oot_{}.tf_all'.format(ver), out_file='score_oot_{}'.format(ver), version=ver)
    print("End")

