import tensorflow as tf
import numpy as np
import time

w = tf.Variable([0.5, 0.5], name="w")

cc=400
cd=100
dc=450
dd=350

probca=tf.multiply(tf.tanh(w[0]),0.5)+0.5
probcb=tf.multiply(tf.tanh(w[1]),0.5)+0.5

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
               
		probca_value=session.run(probca).item()
                probcb_value=session.run(probcb).item()
                
		print("ACooperates with Prob: {a:.3f} BCooperates with Prob: {b:.3f}".format(a=probca_value, b=probcb_value))

