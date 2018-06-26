import tensorflow as tf
import numpy as np
import time

last_a =tf.placeholder(tf.float32)
last_b =tf.placeholder(tf.float32)

a = tf.Variable([0.2,0.6,0.6], name="a")
b = tf.Variable([0.3,0.7,0.3], name="b")


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






def Simulate(InitialState,InitialMA,InitialMB,printState,Dyn,ObsA,ObsB,DTA,DTB):
    # Normal TensorFlow - initialize values, create a session and run the model
     
    state=InitialState
    ma=InitialMA
    mb=InitialMB

    for i in range(1000):
        a,ma=DTA(ObsA(state),ma)
        b,mb=DTB(ObsB(state),mb)
        
        state=Dyn(state,a,b)
        
        printState(state)
        time.sleep(1)
        
    

def ActionPairDyn(s,a,b):
    return {'lasta':a,'lastb':b}

def TransparentObs(s):
    return s

def TitForTatA(o,m):
    return (o['lastb'],m)

def TitForTatB(o,m):
    return (o['lasta'],m)

def printF(s):
    print(s)


def GradDecentDTA(o,m):
    #Currently Storing the policy in global space instead of in m
    #Storing in m would be nicer since it makes the dependencies clear
    
    state={last_a:o['lasta'],last_b:o['lastb']}

    session.run(dua, feed_dict=state)
    probca_value=session.run(probca, feed_dict=state).item()
    
    return (probca_value,m)

def GradDecentDTB(o,m):
    #Currently Storing the policy in global space instead of in m
    #Storing in m would be nicer since it makes the dependencies clear
    
    state={last_a:o['lasta'],last_b:o['lastb']}

    session.run(dub, feed_dict=state)
    probcb_value=session.run(probcb, feed_dict=state).item()
    
    return (probcb_value,m)



#Simulate({'lasta':1,'lastb':0},0,0,printF,ActionPairDyn,TransparentObs,TransparentObs,TitForTatA,TitForTatB)
model = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(model)
    Simulate({'lasta':0.2,'lastb':0.5},0,0,printF,ActionPairDyn,TransparentObs,TransparentObs,GradDecentDTA,GradDecentDTB)






