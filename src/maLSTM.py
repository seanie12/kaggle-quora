import tensorflow as tf


class MA_LSTM(object):
    def __init__(self, vocab_size, embedding_size, hidden_size, seq_length):
        self.vocab = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length

        # input placeholder
        self.input_x_a = tf.placeholder(shape=[None, seq_length], dtype=tf.int32, name="input_x_a")
        self.input_x_b = tf.placeholder(shape=[None, seq_length], dtype=tf.int32, name="input_x_b")
        self.input_y = tf.placeholder(shape=[None], dtype=tf.int32, name="input_y")
        self.batch_size = tf.placeholder(dtype=tf.int32, name="batch_size")
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name="dropout_keep_prob")
        with tf.variable_scope("word_embedding"):
            self.embedding_matrix = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
                name="W", trainable=False)

            self.embedding_chars_a = tf.nn.embedding_lookup(params=self.embedding_matrix, ids=self.input_x_a)
            self.embedding_chars_b = tf.nn.embedding_lookup(params=self.embedding_matrix, ids=self.input_x_b)
            self.pairs = tf.concat([self.embedding_chars_a, self.embedding_chars_b], axis=0)

        with tf.variable_scope("LSTM-layer"):
            cell = tf.nn.rnn_cell.LSTMCell(num_units=self.hidden_size)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.dropout_keep_prob)
            outputs, states = tf.nn.dynamic_rnn(cell, inputs=self.pairs,
                                                swap_memory=True, dtype=tf.float32)
            outputs = tf.transpose(outputs, [1, 0, 2])
            outputs = outputs[-1]
            self.outputs_a = outputs[:self.batch_size]
            self.outputs_b = outputs[self.batch_size:]
            # self.concat = tf.concat([self.outputs_a, self.outputs_b], axis=1)

        with tf.variable_scope("output"):
            self.manhattan_dist = tf.reduce_sum(tf.abs(self.outputs_a - self.outputs_b), axis=1, name="dist")
            self.sim = tf.exp(-self.manhattan_dist, name="sim")
            self.loss = tf.losses.mean_squared_error(labels=self.input_y, predictions=self.sim, name="loss")

        self.predictions = tf.cast(self.sim > 0.5, "float")
        correct_pred = tf.equal(self.predictions, tf.cast(self.input_y, tf.float32))
        self.acc = tf.reduce_mean(tf.cast(correct_pred, "float"))


if __name__ == "__main__":
    model = MA_LSTM(vocab_size=6, embedding_size=10, hidden_size=100, seq_length=6)
    feed_dict = {
        model.input_x_a: [[1, 2, 3, 4, 5, 0], [1, 2, 3, 4, 5, 0]],
        model.input_x_b: [[1, 2, 1, 0, 0, 0], [1, 2, 3, 4, 5, 0]],
        model.input_y: [1, 0],
        model.dropout_keep_prob: 1.0
    }
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
    train_op = optimizer.minimize(model.loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(5000):
        loss, pred, a, b, _ = sess.run([model.loss, model.predictions, model.embedding_matrix, model.pairs, train_op],
                                       feed_dict=feed_dict)
        # print(a)
        print(loss, pred)
