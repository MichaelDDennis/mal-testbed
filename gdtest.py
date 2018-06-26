import tensorflow as tf
import numpy as np
import time

last_a =tf.placeholder(tf.float32)
last_b =tf.placeholder(tf.float32)

a = tf.Variable([0.0,1.0,0.0], name="a")
b = tf.Variable([1.0,0.0,0.0], name="b")

cc=400
cd=100
dc=401
dd=200

probca=tf.multiply(tf.tanh(tf.multiply(a[0],last_a)+tf.multiply(a[1],last_b)+a[2]),0.5)+0.5
probcb=tf.multiply(tf.tanh(tf.multiply(b[0],last_a)+tf.multiply(b[1],last_b)+b[2]),0.5)+0.5

probca2=tf.multiply(tf.tanh(tf.multiply(a[0],probca)+tf.multiply(a[1],probcb)+a[2]),0.5)+0.5
probcb2=tf.multiply(tf.tanh(tf.multiply(b[0],probca)+tf.multiply(b[1],probcb)+b[2]),0.5)+0.5



def getUtilFun(cc,cd,dc,dd,probca,probcb):
    ua = tf.multiply(tf.multiply(probca,  probcb),cc)+tf.multiply(tf.multiply(probca, 1- probcb),cd)+tf.multiply(tf.multiply(1-probca,  probcb),dc)+tf.multiply(tf.multiply(1-probca, 1- probcb),dd)
    ub = tf.multiply(tf.multiply(probca,  probcb),cc)+tf.multiply(tf.multiply(probca, 1- probcb),dc)+tf.multiply(tf.multiply(1-probca,  probcb),cd)+tf.multiply(tf.multiply(1-probca, 1- probcb),dd)
    return (ua,ub)

ua1,ub1=getUtilFun(cc,cd,dc,dd,probca,probcb)
ua2,ub2=getUtilFun(cc,cd,dc,dd,probca2,probcb2)

ua=ua1+ua2
ub=ub1+ub2



# The Gradient Descent Optimizer does the heavy lifting
dua = tf.train.GradientDescentOptimizer(0.01).minimize(0-ua, var_list=[a])
dub = tf.train.GradientDescentOptimizer(0.01).minimize(0-ub, var_list=[b])



# Normal TensorFlow - initialize values, create a session and run the model
model = tf.global_variables_initializer()

memory={last_a:1.0, last_b:1.0}

with tf.Session() as session:
	session.run(model)
	for i in range(1000):
		session.run(dua, feed_dict=memory)
		session.run(dub, feed_dict=memory)
	        time.sleep(1) 
                
		probca_value=session.run(probca, feed_dict=memory).item()
                probcb_value=session.run(probcb, feed_dict=memory).item()
                ua_value=session.run(ua,feed_dict=memory).item()
                ub_value=session.run(ub,feed_dict=memory).item()

                memory[last_a]=probca_value
                memory[last_b]=probcb_value

		print("ACooperates with Prob: {a:.3f} BCooperates with Prob: {b:.3f}".format(a=probca_value, b=probcb_value))
                print("UA: {a:.3f} UB: {b:.3f}".format(a=ua_value,b=ub_value))
