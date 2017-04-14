from __future__ import division, print_function
import tensorflow as tf

from data_process import *
train_phase = True
load_pre = False
prefix = 'att34_13000'
prefix_save = 'att34_rnn1'

#params
lstm_size = 512
timestamp = 1
lamda = 0.001
reg = 0.0005
in_vector_size = 512
in_num_vectors = 14*14

batch_size = 64
n_itr = 50000
save_after = 10000
lr_val = 1e-4

#graph
a = tf.placeholder(tf.float32, [None, in_vector_size, in_num_vectors])
y = tf.placeholder(tf.float32, [None, 8])
nof_y = tf.placeholder(tf.float32, [None, 2])
phase = tf.placeholder(tf.bool)

att_rnn = tf.contrib.rnn.LSTMCell(lstm_size, cell_clip=10.)

lr = tf.placeholder(tf.float32, shape=[])

a_mean = tf.reduce_mean(a, axis=[2])

h_prev = tf.contrib.layers.fully_connected(a_mean, 1024, activation_fn=tf.nn.relu, weights_regularizer = tf.contrib.layers.l2_regularizer(reg))
h_prev = tf.contrib.layers.dropout(h_prev, keep_prob=0.75, is_training=phase)
h_prev = tf.contrib.layers.fully_connected(h_prev, lstm_size, activation_fn=tf.nn.relu, weights_regularizer = tf.contrib.layers.l2_regularizer(reg))

c_prev = tf.contrib.layers.fully_connected(a_mean, 1024, activation_fn=tf.nn.relu, weights_regularizer = tf.contrib.layers.l2_regularizer(reg))
c_prev = tf.contrib.layers.dropout(c_prev, keep_prob=0.75, is_training=phase)
c_prev = tf.contrib.layers.fully_connected(c_prev, lstm_size, activation_fn=tf.nn.relu, weights_regularizer = tf.contrib.layers.l2_regularizer(reg))

nof_logits = tf.contrib.layers.fully_connected(a_mean, 1024, activation_fn=tf.nn.relu, weights_regularizer = tf.contrib.layers.l2_regularizer(reg))
nof_logits = tf.contrib.layers.dropout(nof_logits, keep_prob=0.75, is_training=phase)
nof_logits = tf.contrib.layers.fully_connected(nof_logits, 512, activation_fn=tf.nn.relu, weights_regularizer = tf.contrib.layers.l2_regularizer(reg))
nof_logits = tf.contrib.layers.dropout(nof_logits, keep_prob=0.75, is_training=phase)
nof_logits = tf.contrib.layers.fully_connected(nof_logits, 2, activation_fn=None, weights_regularizer = tf.contrib.layers.l2_regularizer(reg))
nof_log_loss =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=nof_y, logits=nof_logits))

e = tf.contrib.layers.flatten(a)
e = tf.contrib.layers.fully_connected(e, 1024, activation_fn=tf.nn.relu, weights_regularizer = tf.contrib.layers.l2_regularizer(reg))
e = tf.contrib.layers.dropout(e, keep_prob=0.75, is_training=phase)
e = tf.contrib.layers.fully_connected(e, 512, activation_fn=tf.nn.relu, weights_regularizer = tf.contrib.layers.l2_regularizer(reg))
e = tf.contrib.layers.dropout(e, keep_prob=0.75, is_training=phase)
e = tf.contrib.layers.fully_connected(e, in_num_vectors, activation_fn=None, weights_regularizer = tf.contrib.layers.l2_regularizer(reg))

state = (h_prev, c_prev)
h = h_prev
output = [0]*timestamp
alphas = [0]*timestamp

for t in range(timestamp):
    with tf.variable_scope("attention", reuse=(t != 0)):
        beta = tf.contrib.layers.fully_connected(h, 1, activation_fn=tf.nn.relu6, weights_regularizer = tf.contrib.layers.l2_regularizer(reg))

        e_h = tf.contrib.layers.fully_connected(h, 256, activation_fn=tf.nn.relu, weights_regularizer = tf.contrib.layers.l2_regularizer(reg))
        e_h = tf.contrib.layers.dropout(e_h, keep_prob=0.75, is_training=phase)
        e_h = tf.contrib.layers.fully_connected(e_h, in_num_vectors, activation_fn=None, weights_regularizer = tf.contrib.layers.l2_regularizer(reg))

        alpha = tf.nn.softmax(e + e_h)
        alphas[t] = alpha
        z_cap = tf.matmul(a, tf.reshape(alpha, [-1, in_num_vectors, 1]))
        z_cap = tf.reshape(z_cap, [-1, in_vector_size])
        z_cap = tf.multiply(beta, z_cap)
        
        h, state = att_rnn(z_cap, state)

        output[t] = tf.contrib.layers.fully_connected(h, 8, activation_fn=None, weights_regularizer = tf.contrib.layers.l2_regularizer(reg))
    
logits = tf.reduce_mean(output, 0)
log_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
pred = tf.nn.softmax(logits)

alphas_sum_t = tf.reduce_sum(alphas, 1)
alphas_sum_sq = tf.square(1 - alphas_sum_t)
loss = log_loss + lamda*tf.reduce_sum(alphas_sum_sq) + nof_log_loss

optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.5)
train_step = optimizer.minimize(loss)
#train_step = optimizer.apply_gradients(capped_grads_and_vars)


correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# training


if train_phase:
    conv_feat, trn_labels = get_train_data()
    conv_val_feat, val_labels = get_val_data()
    val_nof_y = np.zeros((len(val_labels), 2))
    val_nof_y[np.arange(len(val_labels)), val_labels[:, 4].astype(int)] = 1

    train_acc = 0
    train_log_loss = 0
    train_nof_loss = 0
    train_total_loss = 0
    train_itr = DataIterator(conv_feat, trn_labels)

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    ckpt_file= "%s/models/%s.ckpt"% (path,  prefix)
    import os.path
    if load_pre and os.path.exists(ckpt_file + '.index'):
        saver.restore(sess, ckpt_file)
        print("Model restored from: " + ckpt_file)
        print("========================================================")
    else:
        tf.global_variables_initializer().run()
        print("training from scratch")
        print("========================================================")

    for i in range(n_itr):
        batch_x, batch_y = train_itr.next_batch(batch_size)
        batch_nof_y = np.zeros((batch_size, 2))
        batch_nof_y[np.arange(batch_size), batch_y[:, 4].astype(int)] = 1
    
        l1, l2, l3, _ = sess.run([log_loss, nof_log_loss, loss, train_step], feed_dict={a: batch_x, y:batch_y, nof_y:batch_nof_y, lr:lr_val, phase:True})
        train_log_loss += l1
        train_nof_loss += l2
        train_total_loss += l3
        train_acc += accuracy.eval(feed_dict={a: batch_x, y: batch_y, phase:False})
        if i%100==0:
            if i > 0:
                train_acc = train_acc/100
                train_log_loss = train_log_loss/100
                train_nof_loss = train_log_loss/100
                train_total_loss = train_total_loss/100
            val_accuracy = accuracy.eval(feed_dict={a:conv_val_feat, y: val_labels, phase:False})
            val_log_loss = log_loss.eval(feed_dict={a:conv_val_feat, y: val_labels, phase:False})
            val_nof_loss = nof_log_loss.eval(feed_dict={a:conv_val_feat, y: val_labels, nof_y: val_nof_y, phase:False})
            
            print("iter=%d : loss: (train, val) => log: (%f, %f), nof: (%f, %f) | total train: %f" % 
                (i, train_log_loss, val_log_loss, train_nof_loss, val_nof_loss, train_total_loss ))
            print("train accuracy %g, validation accuracy %g"%(train_acc, val_accuracy))
            train_acc = 0
            train_log_loss = 0
            train_nof_loss  = 0
            train_total_loss = 0
        if i > 0 and i % 1000 == 0:
            lr_val *= 0.9
        if val_accuracy > 0.95:
            save_after = i - 1

        if i > save_after and (i - save_after) % 1500 == 0:
            ckpt_file= "%s/models/%s_%d.ckpt"% (path,  prefix_save, i)
            print("Model saved in file: %s" % saver.save(sess, ckpt_file))

    ckpt_file= "%s/models/%s.ckpt"% (path,  prefix_save)
    print("Model saved in file: %s" % saver.save(sess, ckpt_file))
    sess.close()
else:
    import pandas as pd

    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    ckpt_file= "%s/models/%s.ckpt"% (path,  prefix)

    saver.restore(sess, ckpt_file)
    print("Model restored")

    conv_test_feat, raw_test_filenames = get_test_data()
    subm = pred.eval(feed_dict={a: conv_test_feat, phase:False})

    classes = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    submission = pd.DataFrame(subm, columns=classes)
    submission.insert(0, 'image', raw_test_filenames)
    submission.head()

    subm_name = "%s/submission/%s.gz"% (path,  prefix)
    submission.to_csv(subm_name, index=False, compression='gzip')
    sess.close()
