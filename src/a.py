import tensorflow as tf
import pandas
import numpy as np
from tensorflow.contrib import learn

num_filters = 128
filter_sizes = [3, 4, 5]
num_class = 2
num_epoch = 10
batch_size = 1024
data = pandas.read_csv("../data/train.csv")
question1 = data['question1']
question2 = data['question2']
labels = data['is_duplicate']
embedding_size = 300
test_data = pandas.read_csv("../data/test.csv")
tquestion1 = test_data['question1']
tquestion2 = test_data['question2']
test = []
num_data = len(labels)
dev_ratio = 0.1
dev_idx = -int(num_data * dev_ratio)

for index, q1 in enumerate(tquestion1):
    temp = str(q1) + " " + str(tquestion2[index])
    test.append(temp)

x_data = []
for index, q1 in enumerate(question1):
    temp = str(q1) + " " + str(question2[index])
    x_data.append(temp)

y_labels = np.array(labels)

max_doc_len = max([len(x.split()) for x in x_data])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=max_doc_len)
x = np.array(list(vocab_processor.fit_transform(x_data)))
test = np.array(list(vocab_processor.fit_transform(test)))
x_train = x[:dev_idx]
y_train = y_labels[:dev_idx]
x_dev = x[dev_idx:]
y_dev = y_labels[dev_idx:]

del question1, question2, y_labels

vocab_size = len(vocab_processor.vocabulary_)

x_input = tf.placeholder(shape=[None, max_doc_len], dtype=tf.int32, name="x_input")
y_input = tf.placeholder(shape=[None], dtype=tf.int32, name="y_input")
keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
print(x_train.shape)
with tf.name_scope("embedding"):
    with tf.device("/cpu:0"):
        embedding_matrix = tf.get_variable(shape=[vocab_size, embedding_size],
                                           initializer=tf.contrib.layers.xavier_initializer(),
                                           name="E")
        embedded_chars = tf.nn.embedding_lookup(embedding_matrix, x_input)
        embedded_chars = tf.expand_dims(embedded_chars, -1)

with tf.name_scope("convolve"):
    pooled_output = []
    for filter_size in filter_sizes:
        filter_shape = [filter_size, embedding_size, 1, num_filters]
        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
        conv = tf.nn.conv2d(embedded_chars, filter=W, strides=[1, 1, 1, 1],
                            padding='VALID', name='conv')
        h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
        pooled = tf.nn.max_pool(h, ksize=[1, max_doc_len - filter_size + 1, 1, 1], strides=[1, 1, 1, 1],
                                padding="VALID", name="pooling")
        pooled_output.append(pooled)

num_filter_total = num_filters * len(filter_sizes)
h_pool = tf.concat(pooled_output, 3)
flatten = tf.reshape(h_pool, shape=[-1, num_filter_total])

h_drop = tf.nn.dropout(flatten, keep_prob=keep_prob)
with tf.name_scope("output"):
    W = tf.get_variable(shape=[num_filter_total, num_class], initializer=tf.contrib.layers.xavier_initializer(),
                        name="W")
    b = tf.Variable(tf.constant(0.1, shape=[num_class]), name="b")
    scores = tf.matmul(flatten, W) + b
    predictions = tf.argmax(scores, axis=1)
    predictions = tf.cast(predictions, dtype=tf.int32)

    correct_predictions = tf.equal(predictions, y_input)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    losses = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_input, logits=scores))
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(losses)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=config)

sess.run(tf.global_variables_initializer())
num_train = x_train.shape[0]
total_batch = x_train.shape[0] // batch_size

num_dev = x_dev.shape[0]
dev_batch_nums = int((num_dev - 1) / batch_size) + 1
for epoch in range(num_epoch):
    for i in range(total_batch):
        idx = np.random.choice(num_train, batch_size, replace=True)
        batch_x = x_train[idx, :]
        batch_y = y_train[idx]
        feed_dict = {x_input: batch_x, y_input: batch_y, keep_prob: 0.5}
        _, loss, acc_ = sess.run([optimizer, losses, accuracy], feed_dict=feed_dict)
        print("Epoch : {}, loss : {}, accuracy : {}".format(epoch, loss, acc_))
    dev_loss = []
    dev_acc = []
    for dev_batch_num in range(dev_batch_nums):
        start_idx = batch_size * dev_batch_num
        end_idx = min(num_dev, (dev_batch_num + 1) * batch_size)
        dev_batch_x = x_dev[start_idx: end_idx]
        dev_batch_y = y_dev[start_idx: end_idx]
        loss, acc = sess.run([losses, accuracy], feed_dict={x_input: dev_batch_x, y_input: dev_batch_y, keep_prob: 1.0})
        dev_loss.append(loss)
        dev_acc.append(acc)
    loss = np.mean(dev_loss)
    acc = np.mean(dev_acc)
    print("dev loss : {} and accuracy :{}".format(loss, acc))
del x_train, y_train
print("end of training")
x_test = []
for index, q1 in enumerate(tquestion1):
    temp = str(q1) + " " + str(tquestion2[index])
    x_test.append(temp)
x_test = np.array(list(vocab_processor.fit_transform(x_test)))
preds = []
num_test = x_test.shape[0]
batch_nums = int((num_test - 1) / batch_size) + 1

for batch_num in range(batch_nums):
    start_idx = batch_size * batch_num
    end_idx = min(num_test, (batch_num + 1) * batch_size)
    batch_x = x_test[start_idx: end_idx]
    pred = sess.run(predictions, feed_dict={x_input: batch_x, keep_prob: 1.0})
    preds = np.concatenate([pred, preds], axis=0)

f = open('submission.csv', 'w')
f.write("test_id,is_duplicate\n")
print(preds.shape)
for i, pred in enumerate(preds):
    f.write(str(i) + "," + str(pred) + "\n")
f.close()
