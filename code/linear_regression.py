import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

vectors_set = []
for i in range(1000):
    x1 = np.random.normal(0.0, 1.0)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1,y1])
x_data = [[v[0]] for v in vectors_set]
y_data = [[v[1]] for v in vectors_set]

learning_rate = 0.5
train_step = 4

x = tf.placeholder(tf.float32, [None,1])
y = tf.placeholder(tf.float32, [None,1])

w = tf.Variable(tf.random_normal([1],0,0.1))
b = tf.Variable(tf.zeros([1]))
y_ = tf.add(tf.multiply(x,w), b)

loss = tf.reduce_mean(tf.square(y_-y))
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(train_step):
        l, _ = sess.run([loss, train], feed_dict={x:x_data, y:y_data})      
        print(i,'Loss:',l)
    print('W:',sess.run(w),'B:',sess.run(b))
    plt.plot(x_data,y_data,'ro')
    plt.plot(x_data,x_data*sess.run(w)+sess.run(b))
    plt.show()
