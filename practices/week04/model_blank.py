import functools
import tensorflow as tf
from easydict import EasyDict
""" Original code style: https://github.com/google-research/mixmatch """

class WideResNet():
    
    def __init__(self, hw, n_filters, repeat, n_classes):
        """ 
        Wide Residual Network
        Original paper: https://arxiv.org/abs/1605.07146
        hw: height and width of input images, 
        ex) 28 and 32 for MINST and CIFAR-10,100 respectively
        n_filters: 'k' in original paper
        repeat: 'N' in original paper
        n_classes: Number of classes
    
        """
        self.hw = hw
        self.n_filters = n_filters
        self.repeat = repeat
        self.n_classes = n_classes

    def classifier(self, x, training, **kwargs):
        """
        x: input image
        output: logits that computed by the feed-forward of inputs
        """

        del kwargs
        bn_args = dict(training=training, momentum=0.999)

        def conv_args(k, f):
            return dict(padding='same', 
                        kernel_initializer=tf.random_normal_initializer(stddev=tf.rsqrt(0.5*k*k*f)))

        def residual(x0, filters, stride=1):
            # Implement residual block at here #
            # x = ??
            # x = ??
            # x = ??
            # x = ??
            # x = ??
            # x = ??

            """
            reshape original tensor
            when shapes of input and output of residual tensor are not same.
            """
            if x0.get_shape()[3] != filters or x0.get_shape()[1] != x.get_shape()[1]:
                # Implement changing the shape of input tensor (x0) 
                # when its number of channels is different to that of x #
                # x0 = ??
            return x+x0    

        channels = [16*self.n_filters, 
                    32*self.n_filters, 
                    64*self.n_filters]    
        with tf.variable_scope('classifier', reuse=tf.AUTO_REUSE):
            y = tf.layers.conv2d(x, 16, 3, **conv_args(3, 16))
            for scale in range(len(channels)):
                y = residual(y, channels[scale], stride=2)
                for i in range(self.repeat-1):
                    y = residual(y, channels[scale])
            y = tf.reduce_mean(y, [1,2]) # Global Average Pooling                                
            # logits = ?? <-- Implement the fully-connected layer that returns logits
        return logits            

    def ops_dict(self, wd, **kwargs):
        """
        returns dictionary of operations
        x: input images
        label: input labels
        train_op: training operation
        tune_op: fine-tuning the batch_normalization coeffs
        loss: negative-log-likelihood (NLL) 
        classify: returning softmax-activated output
        """

        hwc = [self.hw, self.hw, 3]
        x_in = tf.placeholder(tf.float32, [None]+hwc, 'x')
        l_in = tf.placeholder(tf.int32, [None], 'labels')
        l_onehot = tf.one_hot(l_in, self.n_classes)

        classifier = functools.partial(self.classifier, **kwargs)
        logits = classifier(x_in, training=True)

        # loss = ??? <-- Implement computing loss function
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        tf.summary.scalar('losses/xe', loss)

        kernels = [v for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
                   if 'kernel' in v.name]
        lr = tf.Variable(0.0, trainable=False)
        # train_op = ??? <-- Implement training operation with the AdamWOptimizer

        train_op = tf.group([train_op, update_ops])

        # Tuning op: only retrain batch norm            
        skip_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        classifier(x_in, training=True)
        train_bn = tf.group(*[v for v in tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                              if v not in skip_ops])

        return EasyDict(
            x=x_in, label=l_in, lr=lr, train_op=train_op, tune_op=train_bn, loss=loss,
            classify = tf.nn.softmax(classifier(x_in, training=False), axis=1)) 
