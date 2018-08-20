import tensorflow as tf
import numpy as np
from cocob_optimizer import COCOB
slim = tf.contrib.slim

class Agent:

    def __init__(self):
#         self.network = self.build_network()
        self.is_training = True
        self.batch_norm_params = {'decay': 0.9,
                            'epsilon': 0.001,
                            'is_training': self.is_training,
                            'scope': 'batch_norm'}

    def predict(self, x, is_training=True):
        # input: [19, 19, 17]
        
        # conv: [19, 19, 256]
        conv = self.conv_layer(x)

        # resnet: [19, 19, 256]
        resnet = self.resnet_layer(x=conv, output_channel=256, n_layer=40)

        # policy: [1, 19*19+1] == [1, 392]
        policy = self.policy_head(resnet)
        
        # value: [1, 1]
        value = self.value_head(resnet)
        
        # [1, 393]
        return tf.concat([policy, value], axis=1)
        
    # def act(self, x):
    #     # return only the policy
    #     return self.predict(x)[:-1]

    def get_policy(self, x):
    	return self.predict(x)[:-1]

    def conv_layer(self, x, output_channel=256, kernel_size=[3,3], is_training=True):
        # input: [19, 19, 17]
        # conv: [19, 19, 256]
        conv = slim.conv2d(x, output_channel, kernel_size=[3, 3], 
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=self.batch_norm_params)
        return conv
        
    def residual_block(self, x, output_channel=256, is_training=True, is_end=False):
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                            normalizer_params=self.batch_norm_params, kernel_size=[3, 3]):
            h1 = slim.conv2d(x, output_channel)
            h2 = slim.conv2d(h1, output_channel, activation_fn=None)
            
        if is_end:
            return h2 + x
        return tf.nn.relu(h2 + x)
    
    def resnet_layer(self, x, output_channel=256, n_layer=40):
        net = x

        for i in range(n_layer):
            with tf.variable_scope('res'+str(i)):
                net = self.residual_block(net, output_channel, is_end=(i==n_layer-1))  
                    
        return net
    
    def policy_head(self, x):
        conv = self.conv_layer(x, output_channel=2, kernel_size=[1, 1])
        conv_flatten = slim.flatten(conv)
        policy = slim.fully_connected(conv_flatten, 19*19+1, activation_fn=None)
        return policy

    def value_head(self, x):
        conv = self.conv_layer(x, output_channel=1, kernel_size=[1, 1])
        conv_flatten = slim.flatten(conv)
        fc = slim.fully_connected(conv_flatten, 256)
        value = slim.fully_connected(fc, 1, activation_fn=tf.nn.tanh)
        return value
    
    def loss(self, x, mcts):
        pass