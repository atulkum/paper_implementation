
def model(a, b):
        self.dropout = tf.placeholder_with_default(1.0, shape=())

        embed_hid = 200
        hidden1 = 100
        
        n_dim = 300
        n_tokens = 15
        with tf.variable_scope('embedding'):
            a = tf.reshape(q, [-1, n_dim])
            a = tf.nn.l2_normalize(a, 0)
            a = slim.fully_connected(tf.nn.dropout(a, keep_prob=self.dropout), embed_hid, activation_fn=tf.nn.relu,
                                     weights_regularizer=slim.l2_regularizer(self.reg))
            a = tf.reshape(a, [-1, n_tokens, embed_hid])

        with tf.variable_scope('embedding', reuse=True):
            b = tf.reshape(b, [-1, n_dim])
            b = tf.nn.l2_normalize(b, 0)
            b = slim.fully_connected(tf.nn.dropout(b, keep_prob=self.dropout), embed_hid, activation_fn=tf.nn.relu,
                                       weights_regularizer=slim.l2_regularizer(self.reg))
            b = tf.reshape(b, [-1, n_tokens, embed_hid])

        e = tf.matmul(a, tf.transpose(b, perm =[0, 2, 1]))
        e_i = tf.nn.softmax(e, dim=-1)
        e_j = tf.nn.softmax(tf.transpose(e, perm=[0, 2, 1]), dim=-1)
        beta = tf.matmul(e_i, b)
        alpha = tf.matmul(e_j, a)

        with tf.variable_scope('G'):
            v = tf.concat([a, beta], axis=2)
            v = tf.reshape(v, [-1, 2 * embed_hid])
            v = slim.fully_connected(tf.nn.dropout(v, keep_prob=self.dropout), hidden1, activation_fn=tf.nn.relu,
                                     weights_regularizer=slim.l2_regularizer(self.reg))
            v1i = tf.reshape(v, [-1, n_tokens, hidden1])
        with tf.variable_scope('G', reuse=True):
            v = tf.concat([b, alpha], axis=2)
            v = tf.reshape(v, [-1, 2 * embed_hid])
            v = slim.fully_connected(tf.nn.dropout(v, keep_prob=self.dropout), hidden1, activation_fn=tf.nn.relu,
                                       weights_regularizer=slim.l2_regularizer(self.reg))
            v2j = tf.reshape(v, [-1, n_tokens, hidden1])

        v1 = tf.reduce_sum(v1i, 1)
        v2 = tf.reduce_sum(v2j, 1)

        with tf.variable_scope('H'):
            y = slim.fully_connected(tf.concat([v1, v2], axis=1), 1, activation_fn=None,
                                       weights_regularizer=slim.l2_regularizer(self.reg))

