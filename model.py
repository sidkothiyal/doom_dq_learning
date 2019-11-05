import tensorflow as tf


class DeepQNetwork:
    def __init__(self, state_size, no_actions, learning_rate, name="DQNetwork"):
        self.state_size = state_size
        self.no_actions = no_actions
        self.learning_rate = learning_rate
        self.name = name
        self.init()

    def init_input(self):
        self.inputs = tf.placeholder(tf.float32, [None, *self.state_size], name="inputs")

    def init_target_q_layer(self):
        self.actions = tf.placeholder(tf.float32, [None, self.no_actions], name="actions")

        self.target_Q = tf.placeholder(tf.float32, [None], name="target")

    def init_conv_layer_1(self):
        self.conv1 = tf.layers.conv2d(inputs=self.inputs,
                                      filters=64,
                                      kernel_size=[8, 8],
                                      strides=[2, 2],
                                      padding="SAME",
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="conv1")

        self.conv1_batchnorm = tf.layers.batch_normalization(self.conv1,
                                                             training=True,
                                                             epsilon=1e-5,
                                                             name="batch_norm1")

        self.conv1_out = tf.nn.elu(self.conv1_batchnorm, name="conv1_out")

    def init_conv_layer_2(self):
        self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                      filters=128,
                                      kernel_size=[4, 4],
                                      strides=[1, 1],
                                      padding="SAME",
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="conv2")

        self.conv2_batchnorm = tf.layers.batch_normalization(self.conv2,
                                                             training=True,
                                                             epsilon=1e-5,
                                                             name="batch_norm2")

        self.conv2_out = tf.nn.elu(self.conv2_batchnorm, name="conv2_out")

    def init_conv_layer_3(self):
        self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                      filters=128,
                                      kernel_size=[4, 4],
                                      strides=[1, 1],
                                      padding="SAME",
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      name="conv3")

        self.conv3_batchnorm = tf.layers.batch_normalization(self.conv3,
                                                             training=True,
                                                             epsilon=1e-5,
                                                             name="batch_norm3")

        self.conv3_out = tf.nn.elu(self.conv3_batchnorm, name="conv3_out")

    def flatten(self):
        self.flatten_layer = tf.layers.flatten(self.conv2_out, name="flatten")

    def init_fc_layer_1(self):
        self.fc = tf.layers.dense(inputs=self.flatten_layer,
                                  units=512,
                                  activation=tf.nn.elu,
                                  kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                  name="fc1")

    def init_fc_layer_2(self):
        self.fc2 = tf.layers.dense(inputs=self.fc,
                                   units=128,
                                   activation=tf.nn.elu,
                                   kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                   name="fc2")

    def init_output(self):
        self.output = tf.layers.dense(inputs=self.fc2,
                                      units=3,
                                      kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                      activation=None,
                                      name="output")

    def init_model(self):
        self.init_conv_layer_1()
        self.init_conv_layer_2()
        #self.init_conv_layer_3()

        self.flatten()

        self.init_fc_layer_1()
        self.init_fc_layer_2()
        self.init_output()

    def init_q_layer(self):
        self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions), axis=1)

    def init_loss_optimizer(self):
        self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q))

        self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

    def init(self):
        self.init_input()

        self.init_target_q_layer()

        self.init_model()


        self.init_q_layer()

        self.init_loss_optimizer()
