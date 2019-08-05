import os
import time
from absl import app
from absl import flags

import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets.cifar10 import load_data
from model import WideResNet

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
FLAGS = flags.FLAGS

def evaluation(sess, x_list, y_list, ops_dict):
    st = time.time()

    n_batches = x_list.shape[0] // FLAGS.batch_size
    if n_batches % FLAGS.batch_size != 0:
        n_batches += 1

    total_loss = 0.0
    y_label = np.empty([0,])
    y_pred = np.empty([0,10])
    for n in range(n_batches):        
       x = x_list[n*FLAGS.batch_size:(n+1)*FLAGS.batch_size]
       y = y_list[n*FLAGS.batch_size:(n+1)*FLAGS.batch_size]
    
       feed_dict = {ops_dict.x:x, ops_dict.label:y}

       loss, pred = sess.run([
           ops_dict.loss, ops_dict.classify], 
           feed_dict=feed_dict)
       total_loss += loss
       y_label = np.concatenate((y_label, y), axis=0)
       y_pred = np.concatenate((y_pred, pred), axis=0)

    total_loss /= n_batches
    et = time.time()
    accuracy = np.mean(np.equal(y_label, np.argmax(y_pred, axis=1)))
    print ("Loss:", round(total_loss,2), 
           "\t Accuracy:", round(accuracy,2),
           "\t Time for evaluation:", round(et-st,2), "(s)")
    return

def train(ops_dict):
    (x_train, y_train), (x_test, y_test) = load_data()
    x_train = np.array(x_train) / 255.0
    x_test = x_test / 255.0
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    n_train = x_train.shape[0] // FLAGS.batch_size
    if n_train % FLAGS.batch_size != 0:
        n_train += 1

    model_name = 'Cifar_10_WRN-'+str(6*FLAGS.repeat+4)+'-'+str(FLAGS.n_filters)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        print ("Start training", model_name)
        sess.run(tf.global_variables_initializer())
        for i in range(FLAGS.num_epoches):
            # assign learning rate
            if i == 0:
                sess.run(tf.assign(ops_dict.lr, FLAGS.init_lr))
            elif i == 60:   
                sess.run(tf.assign(ops_dict.lr, FLAGS.init_lr*0.2))
            elif i == 120:   
                sess.run(tf.assign(ops_dict.lr, FLAGS.init_lr*0.04))
            elif i == 160:   
                sess.run(tf.assign(ops_dict.lr, FLAGS.init_lr*0.008))

            """
            xy_train = list(zip(x_train, y_train))
            random.shuffle(xy_train)
            x_train, y_train = [[x for x, y in xy_train],
                                   [y for x, y in xy_train]]
            x_train = np.asarray(x_train)
            y_train = np.asarray(y_train)                                   
            """

            # Training
            for n in range(n_train):        
               x = x_train[n*FLAGS.batch_size:(n+1)*FLAGS.batch_size]
               y = y_train[n*FLAGS.batch_size:(n+1)*FLAGS.batch_size]
        
               st = time.time()
               feed_dict = {ops_dict.x:x, ops_dict.label:y}
               _ = sess.run(ops_dict.train_op, feed_dict=feed_dict)
               et = time.time()
               print ('Time for ', n,'-th iteration:', (et-st), '(s)')

            # Evaluation               
            print (i, "-th epoch")
            evaluation(sess, x_train, y_train, ops_dict)
            evaluation(sess, x_test, y_test, ops_dict)

            if FLAGS.save_ckpt:
                ckpt_file = os.path.join(FLAGS.save_dir, model_name+'.ckpt')
                saver.save(sess, ckpt_file, global_step=i)
    return

def main(argv):
    del argv
    with tf.Graph().as_default():
        model = WideResNet(
            hw=FLAGS.hw,
            n_filters=FLAGS.n_filters, 
            repeat=FLAGS.repeat, 
            n_classes=FLAGS.n_classes)

        ops_dict = model.ops_dict(FLAGS.wd)
        train(ops_dict)
    return

if __name__ == '__main__':
    flags.DEFINE_integer('n_filters', 10, '')
    flags.DEFINE_integer('hw', 32, 'Number of filters in convolutions.')
    flags.DEFINE_integer('repeat', 4, 'Number of skip-connections in each residual stage.')
    flags.DEFINE_integer('n_classes', 10, 'Number of classes')
    flags.DEFINE_float('init_lr', 1e-2, 'Learning rate.')
    flags.DEFINE_float('wd', 5e-4, 'Weight decay parameter.')
    flags.DEFINE_integer('batch_size', 100, 'Batch size.')
    flags.DEFINE_integer('num_epoches', 200, 'Batch size.')
    flags.DEFINE_integer('keep_ckpt', 5, 'How many checkpoint files will be kept')
    flags.DEFINE_bool('save_ckpt', False, 'Whether to save checkpoint files after finishing every training epoch')
    flags.DEFINE_string('save_dir', './save_ckpt/', 'Directory of summary files.')
    app.run(main)
