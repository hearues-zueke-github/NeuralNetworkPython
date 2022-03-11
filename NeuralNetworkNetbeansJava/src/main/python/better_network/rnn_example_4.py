#! /usr/bin/python3

#Source code with the blog post at http://monik.in/a-noobs-guide-to-implementing-rnn-lstm-using-tensorflow/
import numpy as np
import random
from random import shuffle
import tensorflow as tf

import sys
import select

# from tensorflow.models.rnn import rnn_cell
# from tensorflow.models.rnn import rnn

def generate_sequences(N, T):
    return np.random.randint(0, 2, (N, T, 1)).astype(np.float32)

n_out = 3
def generate_targets_xor(SEQ, offset=5):
    return SEQ[:, :n_out].reshape((-1, n_out))
    # return (SEQ[:, offset, 0].astype(np.uint8) ^ SEQ[:, offset+1, 0].astype(np.uint8)).astype(np.float32).reshape((-1, 1))

NUM_EXAMPLES = 100

train_seq_input = generate_sequences(NUM_EXAMPLES, 20)
train_seq_output = generate_targets_xor(train_seq_input)
print("train_seq_output: {}".format(train_seq_output))

print("Test")
train_input = ['{0:020b}'.format(i) for i in range(2**8)]
shuffle(train_input)
train_input = [map(int,i) for i in train_input]
ti  = []
for i in train_input:
    temp_list = []
    for j in i:
            temp_list.append([j])
    ti.append(np.array(temp_list))
train_input = ti

train_output = []
for i in train_input:
    count = 0
    for j in i:
        if j[0] == 1:
            count+=1
    temp_list = ([0]*21)
    temp_list[count]=1
    train_output.append(temp_list)

test_input = train_input[NUM_EXAMPLES:]
test_output = train_output[NUM_EXAMPLES:]
train_input = train_input[:NUM_EXAMPLES]
train_output = train_output[:NUM_EXAMPLES]

print("test_input.shape: {}".format(np.array(test_input).shape))
print("test_output.shape: {}".format(np.array(test_output).shape))

print("test and training data loaded")

# print("train_input:\n{}".format(train_input))
# print("train_output:\n{}".format(train_output))
# print("test_input:\n{}".format(test_input))
# print("test_output:\n{}".format(test_output))
# input("Press ENTER...")

x = tf.placeholder(tf.float32, [None, 20, 1]) #Number of examples, number of input, dimension of each input
y_ = tf.placeholder(tf.float32, [None, n_out])
# y_ = tf.placeholder(tf.float32, [None, 21])
num_hidden = 24

cell = tf.nn.rnn_cell.LSTMCell(num_hidden)#, state_is_tuple=True)

val, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)
val = tf.transpose(val, [1, 0, 2])
last = tf.gather(val, int(val.get_shape()[0]) - 1)

weight = tf.Variable(tf.truncated_normal([num_hidden, int(y_.get_shape()[1])]))
bias = tf.Variable(tf.constant(0.1, shape=[y_.get_shape()[1]]))

y = tf.nn.softmax(tf.matmul(last, weight) + bias)
cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y,1e-10,1.0)))
minimize = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)
mistakes = tf.not_equal(tf.argmax(y_, 1), tf.argmax(y, 1))
error = tf.reduce_mean(tf.cast(mistakes, tf.float32))

init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)

train_input = train_seq_input
train_output = train_seq_output

test_input = train_seq_input
test_output = train_seq_output

batch_size = 100
no_of_batches = len(train_input) // batch_size
print("len(train_input): {}".format(len(train_input)))
print("no_of_batches: {}".format(no_of_batches))
# input("Press ENTER...")
epoch = 500
errors = []
for i in range(epoch):
    ptr = 0
    for j in range(no_of_batches):
        # print("Im there!")
        inp, out = train_input[ptr:ptr+batch_size], train_output[ptr:ptr+batch_size]
        ptr += batch_size
        sess.run(minimize, {x: inp, y_: out})
    err = sess.run(cross_entropy, {x: test_input, y_: test_output})
    errors.append(err)
    print("epoch: {}, error: {}".format(i, err))
    # User break the loop by pressing ENTER!
    if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
        break
incorrect = sess.run(error, {x: test_input, y_: test_output})
little_input = [[[1],[0],[0],[1],[1],[0],[1],[1],[1],[0],[1],[0],[0],[1],[1],[0],[1],[1],[1],[0]]]
predicts = sess.run(y, {x: little_input})
print(predicts)
print("little_input:\n{}".format(little_input))
print(np.argmax(predicts[0]))
print(20 - np.argmax(predicts[0][::-1]))
print('Epoch {:2d} error {:3.1f}%'.format(i + 1, 100 * incorrect))
sess.close()
