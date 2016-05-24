import tensorflow as tf
import numpy as np

n_digits = 3

def classify(prob):
    max_pred_digits = []
    cum_max_pred = []

    for i in range(n_digits):
        log_prob = tf.log(prob[i])
        max_pred_digits.append(tf.argmax(log_prob,1))
        max_pred = tf.reduce_max(log_prob,1)
        if i == 0:
            cum_max_pred.append(max_pred)
        else:
            cum_max_pred.append(tf.accumulate_n([cum_max_pred[i-1], max_pred]))
    
    max_pred_digits = tf.reshape(tf.concat(0, max_pred_digits), [-1, n_digits])
    
    log_prob_len = tf.log(prob[n_digits])
    log_prob_len = tf.split(1,n_digits+1,log_prob_len)
    
    total_max_pred = []
    total_max_pred.append(log_prob_len[0])

    for i in range(n_digits):
        total_max_pred.append(tf.accumulate_n([log_prob_len[i+1], tf.reshape(cum_max_pred[i], [-1,1])]))
    
    total_max_pred = tf.reshape(tf.concat(0, total_max_pred), [-1, len(total_max_pred)])
    total_len = tf.cast(tf.argmax(total_max_pred,1), tf.int32)
   
    batch_size = total_len.get_shape().as_list()[0]
 
    lengths_transposed = tf.expand_dims(total_len, 1)
    lengths_tiled = tf.tile(lengths_transposed, [1, n_digits])
    
    range_all = tf.range(0, n_digits, 1)
    range_row = tf.expand_dims(range_all, 0)
    range_tiled = tf.tile(range_row, [batch_size, 1])

    mask = tf.less(range_tiled, lengths_tiled)
    all_neg_ones = tf.cast(tf.fill(tf.shape(mask), -1), tf.int64)
	
    result = tf.select(mask, max_pred_digits, all_neg_ones)
    
    return result

x = []
x.append(tf.constant([[0.1, 0.1, 0.8],
			[0.1, 0.1, 0.8],
			[0.7, 0.2, 0.1],
			[0.1, 0.1, 0.8],
			[0.1, 0.1, 0.8]]))
x.append(tf.constant([
			[0.1, 0.1, 0.8],
			[0.1, 0.1, 0.8],
			[0.1, 0.1, 0.8],
			[0.1, 0.1, 0.8],
			[0.1, 0.1, 0.8]
		    ]))
x.append(tf.constant([
			[0.1, 0.1, 0.8],
			[0.1, 0.1, 0.8],
			[0.1, 0.1, 0.8],
			[0.1, 0.1, 0.8],
			[0.1, 0.1, 0.8]
		    ]))
x.append(tf.constant([
			[0.6, 0.1, 0.2, 0.1],
			[0.2, 0.3, 0.4, 0.1],
			[0.2, 0.3, 0.4, 0.1],
			[0.2, 0.3, 0.4, 0.1],
			[0.2, 0.3, 0.4, 0.1]
		    ]))

y = classify(x)



#labels =  [[1,2,3,1,-1], [0,1,0,2,4], [9, 8, 0, 2,3]]
#table = tf.constant(np.identity(10, dtype=np.float32))
#z = tf.nn.embedding_lookup(table, labels, validate_indices=False)


with tf.Session(''):
  print y.eval()

