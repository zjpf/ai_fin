import tensorflow as tf
keras = tf.keras
Lambda = keras.layers.Lambda


class BatchGather(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BatchGather, self).__init__(**kwargs)

    def build(self, input_shape):
        super(BatchGather, self).build(input_shape)

    def call(self, t):
        assert isinstance(t, list)
        sequence_tensor, masked_lm_positions = t
        masked_lm_positions = tf.cast(masked_lm_positions, dtype=tf.int32)
        out_tensor = tf.compat.v1.batch_gather(sequence_tensor, masked_lm_positions)
        return out_tensor

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        shape_a, shape_b = input_shape
        return shape_a[0], shape_b[1], shape_a[2]


class LossLayer(tf.keras.layers.Layer):
    def __init__(self, position_length=None, label_size=None, **kwargs):
        super(LossLayer, self).__init__(**kwargs)
        self.position_length = position_length
        self.label_size = label_size

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(LossLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        assert isinstance(inputs, list)
        log_prob, masked_lm_ids, masked_lm_weights = inputs
        label_ids = tf.reshape(masked_lm_ids, [-1])
        label_weights = tf.reshape(masked_lm_weights, [-1])
        # log_prob = log_prob[:,0,:]
        log_prob = tf.reshape(log_prob, [-1, self.label_size])
        # [batch_size*position_length, label_size]
        label_ids = tf.cast(label_ids, dtype=tf.int32)
        one_hot_labels = tf.one_hot(label_ids, depth=self.label_size, dtype=tf.float32)
        # [batch_size*position_length]
        # per_example_loss = -tf.reduce_sum(log_prob * one_hot_labels, axis=[-1])
        # 支持sigmoid
        per_example_loss = -tf.reduce_sum(50*tf.math.log(log_prob) * one_hot_labels + (1-one_hot_labels)*tf.math.log(1-log_prob),
                                          axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)

        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator
        categorical_accuracy = tf.keras.metrics.categorical_accuracy(one_hot_labels, log_prob)
        top_k_categorical_accuracy = tf.keras.metrics.top_k_categorical_accuracy(one_hot_labels, log_prob)
        log_prob_exp = tf.math.exp(log_prob)
        category_loss = tf.keras.losses.categorical_crossentropy(one_hot_labels,log_prob_exp,from_logits=True)
        log_prob = tf.reshape(log_prob, [-1, self.position_length, self.label_size], name='prediction_layer')
        self.add_loss(loss)
        return log_prob

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return [-1, self.position_length, self.label_size]

    def get_config(self):
        config = {'position_length': self.position_length, 'label_size': self.label_size}
        base_config = super(LossLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class DivConstant(keras.layers.Layer):
    def __init__(self, div_value, **kwargs):
        self.div_value = div_value
        super(DivConstant, self).__init__(**kwargs)

    def build(self, input_shape):
        super(DivConstant, self).build(input_shape)

    def call(self, x):
        out = tf.realdiv(x, self.div_value)
        return out

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'div_value': self.div_value}
        base_config = super(DivConstant, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Dim2Mask(keras.layers.Layer):
    def __init__(self, seq_length, **kwargs):
        self.seq_length = seq_length
        super(Dim2Mask, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Dim2Mask, self).build(input_shape)

    def call(self, x):
        mask_a = tf.cast(tf.reshape(x, [-1, 1, self.seq_length]), tf.float32)
        mask_b = tf.cast(tf.reshape(x, [-1, self.seq_length, 1]), tf.float32)
        attention_mask = mask_a*mask_b
        return attention_mask

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'seq_length': self.seq_length}
        base_config = super(Dim2Mask, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SurvivalLossLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SurvivalLossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(SurvivalLossLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        assert isinstance(inputs, list)
        cvr, delay_lambda, conversion_delay, elapsed_time, conversion_label = inputs
        exp_delay_lambda = tf.exp(delay_lambda)
        pos_loss = (tf.log(cvr+1e-12) + delay_lambda - exp_delay_lambda*conversion_delay)*conversion_label
        neg_loss = tf.log(1 - cvr + cvr*tf.exp(-exp_delay_lambda*elapsed_time) + 1e-12)
        loss = tf.reduce_sum(-(1 - conversion_label)*neg_loss - pos_loss)
        self.add_loss(loss)
        self.add_metric(loss, name="loss_metric", aggregation='mean')
        return cvr, exp_delay_lambda

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[:2]


class LogSurvivalLossLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(LogSurvivalLossLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        assert isinstance(input_shape, list)
        super(LogSurvivalLossLayer, self).build(input_shape)

    def call(self, inputs, mask=None):
        assert isinstance(inputs, list)
        cvr, a, b, d, e, conversion_label = inputs
        a = a + 1e-24
        e = e + 1
        # loss = - tf.log(cvr*(b/a)*tf.pow(conversion_delay/a, b-1)/tf.square(1+tf.pow(conversion_delay/a, b)) + 1e-12)
        loss = -(tf.log(cvr*b+1e-24) + (b-1)*tf.log(d+1e-24) - 2*tf.log(tf.pow(a, b)+tf.pow(d, b)) + b*tf.log(a))
        loss = conversion_label*loss - (1-conversion_label)*tf.log(1-cvr+cvr/(1+tf.pow(e/a, b)) + 1e-24)
        loss = tf.reduce_sum(loss)
        self.add_loss(loss)
        self.add_metric(loss, name="loss_metric", aggregation='mean')
        return cvr, a, b

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[:2]


class PositionEmbedding(tf.keras.layers.Layer):
    """定义位置Embedding，这里的Embedding是可训练的。
    """
    def __init__(
        self,
        input_dim,
        output_dim,
        merge_mode='add',
        custom_position_ids=False,
        **kwargs
    ):
        super(PositionEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.merge_mode = merge_mode
        self.custom_position_ids = custom_position_ids

    def build(self, input_shape):
        super(PositionEmbedding, self).build(input_shape)
        self.embeddings = self.add_weight(
            name='embeddings',
            shape=(self.input_dim, self.output_dim),
            initializer=keras.initializers.TruncatedNormal(stddev=0.02)
        )

    def call(self, inputs):
        """如果custom_position_ids，那么第二个输入为自定义的位置id
        """
        if self.custom_position_ids:
            inputs, position_ids = inputs
            if tf.dtype(position_ids) != 'int32':
                position_ids = tf.cast(position_ids, 'int32')
            pos_embeddings = tf.gather(self.embeddings, position_ids)
        else:
            input_shape = tf.shape(inputs)
            batch_size, seq_len = input_shape[0], input_shape[1]
            pos_embeddings = self.embeddings[:seq_len]
            pos_embeddings = tf.expand_dims(pos_embeddings, 0)
            if self.merge_mode != 'add':
                pos_embeddings = tf.tile(pos_embeddings, [batch_size, 1, 1])

        if self.merge_mode == 'add':
            return inputs + pos_embeddings
        else:
            return tf.concatenate([inputs, pos_embeddings])

    def compute_output_shape(self, input_shape):
        if self.custom_position_ids:
            input_shape = input_shape[0]

        if self.merge_mode == 'add':
            return input_shape
        else:
            return input_shape[:2] + (input_shape[2] + self.output_dim,)

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'merge_mode': self.merge_mode,
            'custom_position_ids': self.custom_position_ids
        }
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class ConstMul(tf.keras.layers.Layer):
    def __init__(self, const_val, dtype=None,*args, **kwargs):
        super(ConstMul, self).__init__(dtype=dtype, **kwargs)
        self.const = const_val

    def call(self, inputs, **kwargs):
        return tf.nn.embedding_lookup(self.const,inputs)

    def build(self, input_shape):
        super(ConstMul, self).build(input_shape)

    def compute_output_shape(self):
        return

    def get_config(self):
        config = {'const_val': self.const}
        base_config = super(ConstMul, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
