import tensorflow as tf
import numpy as np
import time
from random import uniform


#I think the defining nature of the prisoner's dilemma is that dc > cc > dd > cd, so, subject to that linear ordering, we can arbitrarily manipulate those four variables, e.g. by giving a very weak incentive to defect or a huge reward for cooperating


last_a =tf.placeholder(tf.float32)
last_b =tf.placeholder(tf.float32)
#^shouldn't these be binary, or words? (e.g. "cooperate/defect"?)
#(It's also a little sketchy using this as a scalar and key)



#Transition dynamics: in order, the entries are P(c|cc),P(c|cd),P(c|dc), and P(c|dd)

#this will be annoying since I'll have to clip the update to zero if the prob is already 0 or 1, ok
#initialized to TFT
"""
yeeeeah, that's not invertible
a = tf.Variable([1.0,0.0,1.0,0.0], name="a")
b = tf.Variable([1.0,0.0,1.0,0.0], name="b")
"""
a = tf.Variable([.9,.1,.95,.05], name="a")
b = tf.Variable([.9,.1,.95,.05], name="b")


#initialized to be near TFT
"""
a = tf.Variable([1-.1*random(),.1*random(),1-.1*random(),random()], name="a") 
b = tf.Variable([1-.1*random(),.1*random(),1-.1*random(),random()], name="b")
"""

#initialized randomly
"""
a = tf.Variable([random(),random(),random(),random()], name="a") 
a = tf.Variable([random(),random(),random(),random()], name="b") 
"""

#alternatively I can do Michael's logistic thing
#initialized to TFT...supposedly (although I feel like you should have a -1 or +1 depending on opponent's action?)
#also shouldn't there only be five parameters? (depending on if last move was cc,cd,dc,dd, plus the bias term)
"""
a = tf.Variable([0.0,1.0,0.0], name="a")
b = tf.Variable([1.0,0.0,0.0], name="b")
"""

#Payoff "matrix" (honestly this is probably better as a vector for exact return calcs?)
#e.g. cd is the return I get when I cooperate and you defect
cc=400.0
cd=100.0
dc=401.0
dd=200.0
#huh, so how would I handle general games? ideally without having to write an entire transition matrix...



#testing basic ops since I'm a numpy noob >.>
#print 5*np.identity(4)




#params: a and b are the policies/transition matrices, gamma is the discount factor
#returns vector v with v_xy the value of that state for A (e.g. v_cd is the value of that state for A after both players cooperated)
#so to get values for b swap cd with dc and a with b
def exactDiscountedValue(cc,cd,dc,dd,a,b):
        #first construct the transition matrix (so T_ij = P(i|j), where i could be cd for example)
        T=[[a[0]*b[0], a[1]*b[2], a[2]*b[1], a[3]*b[3]],
        			  [a[0]*(1-b[0]), a[1]*(1-b[2]), a[2]*(1-b[1]), a[3]*(1-b[3])],
        			  [(1-a[0])*b[0], (1-a[1])*b[2], (1-a[2])*b[1], (1-a[3])*b[3]],
        			  [(1-a[0])*(1-b[0]), (1-a[1])*(1-b[2]), (1-a[2])*(1-b[1]), (1-a[3])*(1-b[3])]]

        #T = tf.convert_to_tensor(np.matrix(M)) 
        print(T)
        #print(T.shape, T.dtype)
        #^not sure if python is going to throw a hissy fit about spacing
		#next construct the reward vector associated with the transitions/new states
        r = ([[cc],[cd],[dc],[dd]])

        #finally solve the recursion for v (i.e. solve v = gammaT^tv+T^tr)
        #v = np.solve((np.identity(3)-gamma*np.transpose(T)),np.matmul(np.tranpose(T),r)) #nooot sure if this solution is differentiable -- might need to use matrix inverse... also might divide by zero T.T
        print tf.linalg.inv(T)

        """
        v = tf.matmul(tf.linalg.inv(1.0*tf.identity(4)
        	-tf.scalar_mul(gamma,tf.transpose(T))),
        tf.matmul(tf.transpose(T),
        	r))
        	"""
        tosee = tf.diag([1.0,1.0,1.0,1.0])-tf.multiply(tf.transpose(T),.99)
        #print session.run(tosee)


        v = tf.matmul(tf.linalg.inv(tosee),
        tf.matmul(tf.transpose(T),r))
       
#^bleh, lame error with float vs. int types that I don't get



        return v



"""
#hacky ways to approximate return

#this really doesn't use all of a or b :p
#yeah, no clue what this is doing...
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

"""

#returns P(c|last_a,last_b) given policy a
def indexIntoA(last_a,last_b):
	return int((2*(1-last_a)+(1-last_b)))


#the LOLA utility will be something like uaLOLA = ua(probca, probcb+grad stuff)-- so I need to get my hands on grad stuff
#also, for more complicated state histories, I guess we can do a linear combination of the last k moves (and then tanh, sure)... 
#although starting the game will be awkward, on well. 
#Might as well ask the OpenAI people for their parametrization when I call them out on their math totally clashing with what they say they're doing

#we'll also want to implement some other strategies, like mine and Michael's versions of LOLA/reward shaping, 
#and Mike Littman's Q-learning thing (which I think is not gradient based? Might not be clear how to translate that into tensor flow... 
#and how the LOLA strategies interact with it)





# Normal TensorFlow - initialize values, create a session and run the model
model = tf.global_variables_initializer()

#initialize to both players cooperating on the zeroth turn
memory={last_a:1.0, last_b:1.0}

#get values for the current states
#gamma = 1.0 #discount factor
va = exactDiscountedValue(cc,cd,dc,dd,a,b)
vb = exactDiscountedValue(cc,cd,dc,dd,b,a)
ua = va[indexIntoA(memory[last_a],memory[last_b])]
ub = vb[indexIntoA(memory[last_b],memory[last_a])]


# The Gradient Descent Optimizer does the heavy lifting
#I take it that dua is the updated set of parameters for player A
dua = tf.train.GradientDescentOptimizer(0.01).minimize(0-ua, var_list=[a])
dub = tf.train.GradientDescentOptimizer(0.01).minimize(0-ub, var_list=[b])

with tf.Session() as session:
        session.run(model)
        for i in range(1000):

        		
                session.run(dua, feed_dict=memory) #I think this updates a?
                session.run(dub, feed_dict=memory)
                print memory
                a_value = session.run(a, feed_dict=memory)
                b_value = session.run(b, feed_dict=memory)
                if (a_value[indexIntoA(memory[last_a],memory[last_b])].item() > uniform(0,1)):
                	last_a_temp = 1
                else:
                	last_a_temp = 0
                if (b_value[indexIntoA(memory[last_b],memory[last_a])].item() > uniform(0,1)):
                	memory[last_b] = 1
                else: 
                	memory[last_b] = 0
                memory[last_a] = last_a_temp #not sure if I'm addressing correctly >.<
                time.sleep(1)

                
                """

                probca_value=session.run(probca, feed_dict=memory).item()
                probcb_value=session.run(probcb, feed_dict=memory).item()
                ua_value=session.run(ua,feed_dict=memory).item()
                ub_value=session.run(ub,feed_dict=memory).item()

                memory[last_a]=probca_value
                memory[last_b]=probcb_value

                #not sure why/how the next two lines print what they say they print
                print("ACooperates with Prob: {a:.3f} BCooperates with Prob: {b:.3f}".format(a=probca_value, b=probcb_value))
                print("UA: {a:.3f} UB: {b:.3f}".format(a=ua_value,b=ub_value))
				"""





