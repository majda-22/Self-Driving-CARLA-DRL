'''
L'architecture de DQNetwork
'''
class DQNetwork():
    def __init__(self, state_size, action_size, learning_rate, name='DQNetwork'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.possible_actions = np.identity(self.action_size, dtype=int).tolist()
        
        with tf.variable_scope(name):
            #inputs define image fed into NN
            self.inputs_ = tf.placeholder(tf.float32, [None, *state_size], name="inputs")
            
            #actions define array containing tuple of actions taken by system
            self.actions_ = tf.placeholder(tf.float32, [None, self.action_size], name="actions_")
            
            
            self.target_Q = tf.placeholder(tf.float32, [None], name="target") 
            
            #multiple convolutions to downsample image and eventually
            #encode it to vector form to feed into neural network
            
            #Output size : 27 x 27 x 64
            self.conv1 = tf.layers.conv2d(inputs=self.inputs_, 
                                          filters=64,
                                          kernel_size=[6,6],
                                          strides=[3,3],
                                          padding="VALID",
                                          kernel_initializer=tf.variance_scaling_initializer(),
                                          name="conv1")
            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")
            
            #Output size: 9 x 9 x 128
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out, 
                                          filters=128,
                                          kernel_size=[3,3],
                                          strides=[3,3],
                                          padding="VALID",
                                          kernel_initializer=tf.variance_scaling_initializer(),
                                          name="conv2")
            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

            #Output size: 6 x 6 x 128
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out, 
                                          filters=128,
                                          kernel_size=[4,4],
                                          strides=[1,1],
                                          padding="VALID",
                                          kernel_initializer=tf.variance_scaling_initializer(),
                                          name="conv3")
            #After multiple convolutions, use exponential linear unit
            #Activation function since DQN predicts continuous set of q-vals
            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")
            
            # self.flatten = tf.contrib.layers.flatten(self.conv3_out)
            self.flatten = Flatten()(self.conv3_out)
            
            self.fc = tf.layers.dense(inputs=self.flatten,
                                      units=512,
                                      activation=tf.nn.elu,
                                      kernel_initializer=tf.variance_scaling_initializer(),
                                      name="fc1")
            
            #force output to number of possible actions so each action has q-val
            self.output = tf.layers.dense(inputs=self.fc,
                                          kernel_initializer=tf.variance_scaling_initializer(),
                                          units=self.action_size,
                                          activation=None,
                                          name="output")
            
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_)) #predicted Q-value computed by DNN by associating output of DNN w/ action tuples
            self.loss = tf.reduce_mean(tf.square(self.target_Q - self.Q)) #compute loss per each action-val 
            
            #return gradients for each weight of NN (change in weights after minimizing loss)
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        
            

    def predict_action(self, sess, explore_start, explore_stop, decay_rate, decay_step, state):
        #Implement epsilon-greedy action selection as policy where majority
        #of time optimal (greedy) action chosen and sometimes random action
        #chosen, probability of choosing randomly defined by explore_prob
        exp_tradeoff = np.random.rand()
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

        if (explore_probability > exp_tradeoff):
            action_int = np.random.choice(self.action_size)
            action = self.possible_actions[action_int]
        else:
            #optimal action selection using output from DQN
            actionQvals = sess.run(self.output, feed_dict={self.inputs_: state.reshape((1, *state.shape))})
            action_int = np.argmax(actionQvals)
            action = self.possible_actions[int(action_int)]

        return action_int, action, explore_probability

