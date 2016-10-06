## IMPLEMENT REGRESSION

# define inputs
x = tf.placeholder(tf.float32, [None, 2])

# define variables
W = tf.Variable(tf.zeros([2, 1]))
b = tf.Variable(tf.zeros([1]))

# define model
y = tf.nn.sigmoid(tf.matmul(x, W) + b)

## TRAINING

# implement cross-entropy
y_ = tf.placeholder(tf.float32, [None, 1])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


# gradient descent
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.initialize_all_variables()
