""" Original codes: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/mnist """

import os

from absl import app
from absl import flags

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

FLAGS = flags.FLAGS

def get_mnist_dataset():
    mnist = input_data.read_data_sets("./mnist/", one_hot=False)
    train_set = mnist.train
    valid_set = mnist.validation
    test_set = mnist.test
    return train_set, valid_set, test_set

def mlp(x, dim, num_classes):
    with tf.name_scope('mlp'):
        h = tf.layers.dense(x, 
                            units=dim,
                            activation=tf.nn.elu,
                            use_bias=True)
        h = tf.layers.dense(x, 
                            units=dim,
                            activation=tf.nn.elu,
                            use_bias=True)
        logits = tf.layers.dense(x, 
                                 units=num_classes,
                                 activation=None,
                                 use_bias=True)
    return logits

def evaluation(logits, labels):
    correct = tf.nn.in_top_k(logits, labels, 1)
    return tf.reduce_sum(tf.cast(correct, tf.int32))    

def do_eval(sess, eval_correct, x, y, ds):
    true_count = 0
    total_steps = ds.num_examples // FLAGS.batch_size
    num_examples = total_steps * FLAGS.batch_size
    for step in range(total_steps):
        images_feed, labels_feed = ds.next_batch(FLAGS.batch_size)
        feed_dict = {x: images_feed, y: labels_feed}    
        true_count += sess.run(eval_correct, feed_dict=feed_dict)
    precision = true_count / num_examples
    print ("Num examples:", num_examples, "Num correct:", true_count, "Precision:", precision)

def run_training():
    train_set, valid_set, test_set = get_mnist_dataset()
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32, shape=[None, 784])
        y = tf.placeholder(tf.int32, shape=[None])

        logits = mlp(x, FLAGS.hidden_dim, 10)
        labels = tf.to_int64(y) 
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        optimizer = tf.train.AdamOptimizer(FLAGS.lr)
        train_op = optimizer.minimize(loss)
        eval_correct = evaluation(logits, y)

        saver = tf.train.Saver()
        init = tf.global_variables_initializer()
        sess = tf.compat.v1.Session()
        sess.run(init)

        print ("Start training")
        for step in range(FLAGS.max_steps):
            images_feed, labels_feed = train_set.next_batch(FLAGS.batch_size)
            feed_dict = {x: images_feed, y: labels_feed}
            _, loss_val = sess.run([train_op, loss], 
                                   feed_dict = feed_dict)

            if (step+1) % FLAGS.eval_step == 0 or (step+1) == FLAGS.max_steps:
                ckpt_file = os.path.join(FLAGS.save_dir, 'model.ckpt')
                saver.save(sess, ckpt_file, global_step=step)

                print ("Training / Validation / Test @ ", (step//FLAGS.eval_step)+1, "-th evaluation")
                do_eval(sess, eval_correct, x, y, train_set)
                do_eval(sess, eval_correct, x, y, valid_set)
                do_eval(sess, eval_correct, x, y, test_set)
    return

def main(argv):
    del argv
    run_training()   
    return

if __name__ == '__main__':
    flags.DEFINE_integer('hidden_dim', 256, 'Dimension of hidden layers.')
    flags.DEFINE_float('lr', 1e-3, 'Learning rate.')
    flags.DEFINE_integer('max_steps', 200, 'Maximum number of iteration steps.')
    flags.DEFINE_integer('eval_step', 10, 'Evaluation at every this step.')
    flags.DEFINE_integer('batch_size', 100, 'Batch size.')
    flags.DEFINE_string('save_dir', './save_ckpt', 'Where to save checkpoint files')
    app.run(main)
