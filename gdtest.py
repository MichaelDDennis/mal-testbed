import tensorflow as tf
import numpy as np
import time

# x and y are placeholders for our training data
x = tf.placeholder("float")
y = tf.placeholder("float")
# w is the variable storing our values. It is initialised with starting "guesses"
# w[0] is the "a" in our equation, w[1] is the "b"
w = tf.Variable([0.5, 0.5], name="w")
# Our model of y = a*x + b

cc=400
cd=100
dc=450
dd=350

probca=tf.multiply(tf.tanh(w[0]),0.5)+0.5
probcb=tf.multiply(tf.tanh(w[1]),0.5)+0.5

mua = tf.multiply(tf.multiply(probca,  probcb),cc)+tf.multiply(tf.multiply(probca, 1- probcb),cd)+tf.multiply(tf.multiply(1-probca,  probcb),dc)+tf.multiply(tf.multiply(1-probca, 1- probcb),dd)
mub = tf.multiply(tf.multiply(probca,  probcb),cc)+tf.multiply(tf.multiply(probca, 1- probcb),dc)+tf.multiply(tf.multiply(1-probca,  probcb),cd)+tf.multiply(tf.multiply(1-probca, 1- probcb),dd)

ua = tf.multiply(tf.multiply(probca,  probcb),cc)+tf.multiply(tf.multiply(probca, 1- probcb),cd)+tf.multiply(tf.multiply(1-probca,  probcb),dc)+tf.multiply(tf.multiply(1-probca, 1- probcb),dd)
ub = tf.multiply(tf.multiply(probca,  probcb),cc)+tf.multiply(tf.multiply(probca, 1- probcb),dc)+tf.multiply(tf.multiply(1-probca,  probcb),cd)+tf.multiply(tf.multiply(1-probca, 1- probcb),dd)

# The Gradient Descent Optimizer does the heavy lifting
dua = tf.train.GradientDescentOptimizer(0.01).minimize(ua)
dub = tf.train.GradientDescentOptimizer(0.01).minimize(ub)

# Normal TensorFlow - initialize values, create a session and run the model
model = tf.global_variables_initializer()

with tf.Session() as session:
	session.run(model)
	for i in range(1000):
		session.run(dua, feed_dict={})
		session.run(dub, feed_dict={})
	        time.sleep(1) 
                #w_value=session.run(w)
                probca_value=session.run(probca).item()
                probcb_value=session.run(probcb).item()
               # print(type(probca_value))
                #print("ACooperates with Prob: {a:.3f} BCooperates with Prob: {b:.3f}".format(a=w_value[0], b=w_value[1]))
                print("ACooperates with Prob: {a:.3f} BCooperates with Prob: {b:.3f}".format(a=probca_value, b=probcb_value))

