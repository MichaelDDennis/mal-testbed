import tensorflow as tf
import numpy as np
import time

a = tf.Variable([0.5], name="a")
b = tf.Variable([0.5], name="b")

cc=400
cd=100
dc=450
dd=350

probca=tf.multiply(tf.tanh(a[0]),0.5)+0.5
probcb=tf.multiply(tf.tanh(b[0]),0.5)+0.5

ua = tf.multiply(tf.multiply(probca,  probcb),cc)+tf.multiply(tf.multiply(probca, 1- probcb),cd)+tf.multiply(tf.multiply(1-probca,  probcb),dc)+tf.multiply(tf.multiply(1-probca, 1- probcb),dd)
ub = tf.multiply(tf.multiply(probca,  probcb),cc)+tf.multiply(tf.multiply(probca, 1- probcb),dc)+tf.multiply(tf.multiply(1-probca,  probcb),cd)+tf.multiply(tf.multiply(1-probca, 1- probcb),dd)

# The Gradient Descent Optimizer does the heavy lifting
dua = tf.train.GradientDescentOptimizer(0.01).minimize(0-ua, var_list=[a])
dub = tf.train.GradientDescentOptimizer(0.01).minimize(0-ub, var_list=[b])

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

