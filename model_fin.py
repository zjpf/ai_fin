# 特征：事件类型、事件时间、授信/动支额度、机构类型/担保类型/业务类型/账户类型
# 授信有效时间、还款期数/还款频率、累计逾期金额、账单金额/实际还款金额/余额
from transformer import Transformer
import tensorflow as tf
from bert_layer import Dim2Mask
from tensorflow.python.keras.utils.vis_utils import plot_model

keras = tf.keras
Embedding = keras.layers.Embedding
Input = keras.Input
Reshape = keras.layers.Reshape
Concatenate = keras.layers.Concatenate
BatchNormalization = keras.layers.BatchNormalization
Add = keras.layers.Add
Multiply = keras.layers.Multiply
Dropout = keras.layers.Dropout
Dense = keras.layers.Dense
LayerNormalization = keras.layers.LayerNormalization
Model = keras.models.Model
Lambda = tf.keras.layers.Lambda


class HeteroTransformer(object):
    def __init__(self, seq_length, fn_voc_size, dnn_dim, num_hidden_layers, num_attention_heads, size_per_head):
        hidden_size = num_attention_heads*size_per_head
        transformer = Transformer(seq_length=seq_length, num_hidden_layers=num_hidden_layers,
                                  num_attention_heads=num_attention_heads, size_per_head=size_per_head)
        initializer = transformer.create_initializer
        input_list = []
        f_seq_length = Input(shape=(1,), name='seq_length', dtype='int32')
        input_list.append(f_seq_length)
        input_mask = Lambda(lambda x: tf.reshape(tf.sequence_mask(tf.squeeze(x), maxlen=seq_length), shape=[-1, seq_length]))(f_seq_length)
        f_dense_mask = Input(shape=(seq_length,), name="mask", dtype='int32')
        input_list.append(f_dense_mask)
        # Lambda(lambda x: tf.cast(tf.reshape(x, [-1, seq_length]), tf.int32))(input_mask)

        # 创建Embedding
        f_embeddings = []    # 忽略规模 [opening_price, closing_price, max_price, min_price：相对当前时刻价格, volume：相对比例]
        for fn in ['event_type']:  # mask同时作为event_type, 'event_dt'
            f_cate_emb = Embedding(input_dim=2, output_dim=hidden_size, embeddings_initializer=initializer())(f_dense_mask)
            f_embeddings.append(f_cate_emb)
        for fn in ['event_dt']:
            f_cate = Input(shape=(seq_length,), name=fn, dtype='int32')
            input_list.append(f_cate)
            f_cate_emb = Embedding(input_dim=seq_length+1, output_dim=hidden_size, embeddings_initializer=initializer())(f_cate)
            f_embeddings.append(f_cate_emb)
        for fn in ['opening_price', 'closing_price', 'max_price', 'min_price', 'volume', 'amount']:
            f_dense = Input(shape=(seq_length,), name=fn, dtype='float32')
            input_list.append(f_dense)
            f_dense_emb = Embedding(input_dim=2, output_dim=hidden_size, embeddings_initializer=initializer())(f_dense_mask)
            dense_mask = Lambda(lambda x: tf.reshape(tf.cast(x, dtype=tf.float32), shape=[-1, seq_length, 1]))(f_dense_mask)
            f_dense_reshape = Lambda(lambda x: tf.reshape(x, shape=[-1, seq_length, 1]))(f_dense)
            f_tanh = Lambda(lambda x: tf.tanh(x))(Multiply()([f_dense_emb, dense_mask, f_dense_reshape]))
            f_embeddings.append(f_tanh)

        emb_out = LayerNormalization()(Add()(f_embeddings))
        attention_mask = Dim2Mask(seq_length)(input_mask)
        transformer_output = transformer.transform(emb_out, attention_mask, get_first=True)
        transformer_output = Lambda(lambda x: tf.reshape(x, [-1, hidden_size]))(transformer_output)
        dnn_out = Dense(dnn_dim, activation='relu', kernel_initializer=initializer(), name="dnn")(transformer_output)
        predict_out = Dense(4, activation='sigmoid', kernel_initializer=initializer(), name="cls")(dnn_out)
        self.model = Model(inputs=input_list, outputs=Reshape((4,))(predict_out))
        self.inputs = input_list

    @staticmethod
    def get_custom_objects():
        return {"gelu": Transformer.gelu, "Dim2Mask": Dim2Mask}


def main():
    # 1. 模型训练
    model = HeteroTransformer(seq_length=100, fn_voc_size=None, dnn_dim=16,
                              num_hidden_layers=2, num_attention_heads=2, size_per_head=8).model
    plot_model(model, to_file="event_model.png", show_shapes=False)
    print("END")


if __name__ == "__main__":
    main()

