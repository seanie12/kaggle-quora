import tensorflow as tf
from src.cnn import CNN
from src import utils
import os
import numpy as np
from gensim.models import KeyedVectors

batch_size = 64
embedding_size = 300
hidden_size = 50
learning_rate = 1e-4
max_grad_norm = 5.0
dropout_keep_prob = 0.5
num_epochs = 1000
filter_sizes = [3, 4, 5]
dev_ratio = 0.1
# load data and labels
input_a, input_b, labels, vocab = utils.load_data_labels("../data/train.csv", train=True)

dev_idx = - int(len(labels) * dev_ratio)
train_input_a = input_a[:dev_idx]
train_input_b = input_b[:dev_idx]
train_labels = labels[:dev_idx]

dev_input_a = input_a[dev_idx:]
dev_input_b = input_b[dev_idx:]
dev_labels = labels[dev_idx:]

vocab_size = len(vocab)
seq_length = input_a.shape[1]

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        print("load the model")
        model = CNN(vocab_size=vocab_size, embedding_size=embedding_size, num_filters=128,
                    seq_length=seq_length, filter_sizes=filter_sizes)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        tvars = tf.trainable_variables()
        grads, global_norm = tf.clip_by_global_norm(tf.gradients(model.loss, tvars), max_grad_norm)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.apply_gradients(zip(grads, tvars), name="train_op", global_step=global_step)
        checkpoint_dir = "../saved/CNN/checkpoints"
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

            # load pre-trained word2vec
            initW = np.random.uniform(-0.25, 0.25, (len(vocab), embedding_size))
            word2vec = KeyedVectors.load_word2vec_format("../data/GoogleNews-vectors-negative300.bin", binary=True)
            for word, idx in vocab.items():
                try:
                    vec = word2vec[word]
                    initW[idx] = vec
                # if there is no pre-trained word vector, use random number as first initialized
                except KeyError:
                    continue
            # zero vector for padding token
            initW[0] = 0.0
            model.embedding_matrix.assign(initW)
            del initW, word2vec

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        sess.run(tf.global_variables_initializer())

        checkpoint_prefix = os.path.abspath(os.path.join(checkpoint_dir, "CNN"))


        def train_step(x_batch_a, x_batch_b, y_batch, dropout_keep_prob):
            feed_dict = {
                model.input_x_a: x_batch_a,
                model.input_x_b: x_batch_b,
                model.input_y: y_batch,
                model.dropout_keep_prob: dropout_keep_prob,
                model.batch_size: len(y_batch)
            }
            _, step, loss, acc = sess.run([train_op, global_step, model.loss, model.acc], feed_dict=feed_dict)
            print("step : {}, loss : {}, acc : {}".format(step, loss, acc))


        def dev_step(x_batch_a, x_batch_b, y_batch):
            feed_dict = {
                model.input_x_a: x_batch_a,
                model.input_x_b: x_batch_b,
                model.input_y: y_batch,
                model.dropout_keep_prob: 1.0,
                model.batch_size: len(y_batch)
            }
            loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)

            return loss, acc


        batches = utils.batch_iter(list(zip(input_a, input_b, labels)), num_epochs=num_epochs,
                                   batch_size=batch_size)
        for batch in batches:
            x_batch_a, x_batch_b, y_batch = zip(*batch)
            train_step(x_batch_a, x_batch_b, y_batch, dropout_keep_prob)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % 100 == 0:
                path = saver.save(sess, checkpoint_prefix, global_step)
                print("save model to {}".format(path))
                dev_loss = []
                dev_acc = []
                dev_batches = utils.batch_iter(list(zip(dev_input_a, dev_input_b, dev_labels)), num_epochs=1,
                                               batch_size=batch_size)
                for dev_batch in dev_batches:
                    x_dev_batch_a, x_dev_batch_b, y_dev_batch = zip(*dev_batch)
                    loss, acc = dev_step(x_dev_batch_a, x_dev_batch_b, y_dev_batch)
                    dev_loss.append(loss)
                    dev_acc.append(acc)
                loss = np.mean(dev_loss)
                acc = np.mean(dev_acc)
                print("dev loss : {}, dev acc : {}".format(loss, acc))
