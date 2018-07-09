import tensorflow as tf
import numpy as np
import time
from random import uniform
import math

def sigmoid(x):
  return 1 / (1 + math.exp(-x))


#I think the defining nature of the prisoner's dilemma is that dc > cc > dd > cd and 2cc>dc+cd,
#so, subject to those constraints we can arbitrarily manipulate those four variables, e.g. by giving a very weak incentive to defect or a huge reward for cooperating


#ask Michael how to use placeholders to time the udpates properly
last_a = tf.placeholder(tf.float32)
last_b = tf.placeholder(tf.float32)
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
a = tf.Variable([1.0,-1.0,1.0,-1.0], name="a")
b = tf.Variable([1.0,-1.0,1.0,-1.0], name="b")

#ok, cool! initializing to TFT is stable :) it seems.
#I guess I'd like a graph of the phase plane
#ugh but it's eight dimensionaaaaal
#also, part of me still feels like I can do this analytically (certainly without simulations)


#initialized to be near TFT
"""
a = tf.Variable([1-.1*uniform(0,1),.1*uniform(0,1),1-.1*uniform(0,1),.1*uniform(0,1)], name="a") 
b = tf.Variable([1-.1*uniform(0,1),.1*uniform(0,1),1-.1*uniform(0,1),.1*uniform(0,1)], name="b")
"""

#initialized randomly
"""
a = tf.Variable([uniform(-1,1),uniform(-1,1),uniform(-1,1),uniform(-1,1)], name="a") 
b = tf.Variable([uniform(-1,1),uniform(-1,1),uniform(-1,1),uniform(-1,1)], name="b") 
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
cc=4.0
cd=2.0
dc=4.2
dd=2.2
#huh, so how would I handle general games? ideally without having to write an entire transition matrix...


"""
#test payoff matrix to check for bugs preventing obvious optimization
cc=4.0
cd=3.0
dc=2.0
dd=1.0
#uuuuuugh, so they're not learning to cooperate, which means there's a bug somewhere >.<
"""


#clip the probabilities of cooperating to *actually* be probabilities
def constrainToProbabilities(a):
	for i in range(4): #I don't know tensorflow for length, ugh T.T
		if a[i] > 1: #I wouldn't be surprised if there's some pass-by-value I'm missing here
			a[i] = 1
		elif a[i] < 0:
			a[i] = 0



#params: a and b are the policies/transition matrices, gamma is the discount factor
#returns vector v with v_xy the value of that state for A (e.g. v_cd is the value of that state for A after both players cooperated)
#so to get values for b swap cd with dc and a with b
#if we wanted no discounting, then we would solve for the stationary vector
# see "Iterated Prisoner's Dilemma contains strategies that dominate any evolutionary opponent"
def exactDiscountedValue(cc,cd,dc,dd,a_pol,b_pol):
	#woooow, so I think the bug is that Tensforflow's gradient thing gives non probabilities before consulting eDV. Great.
    #constrainToProbabilities(a)
    #constrainToProbabilities(b)
    #sigmoid all the entries because apparently gradient optimizer loves illegal updates ugh    
    a = tf.sigmoid(a_pol)
    b = tf.sigmoid(b_pol)
    #first construct the transition matrix (so T_ij = P(i|j), where i could be cd for example)
    T=[[a[0]*b[0], a[1]*b[2], a[2]*b[1], a[3]*b[3]],
	[a[0]*(1-b[0]), a[1]*(1-b[2]), a[2]*(1-b[1]), a[3]*(1-b[3])],
	[(1-a[0])*b[0], (1-a[1])*b[2], (1-a[2])*b[1], (1-a[3])*b[3]],
	[(1-a[0])*(1-b[0]), (1-a[1])*(1-b[2]), (1-a[2])*(1-b[1]), (1-a[3])*(1-b[3])]]
	


    #print(T)#apparently this only prints once? I'm so confused >.>
    #print(T.shape, T.dtype)

	#next construct the reward vector associated with the transitions/new states
    r = ([[cc],[cd],[dc],[dd]])

    #finally solve the recursion for v (i.e. solve v = gammaT^tv+T^tr)
    #v = np.solve((np.identity(3)-gamma*np.transpose(T)),np.matmul(np.tranpose(T),r)) #nooot sure if this solution is differentiable -- might need to use matrix inverse... also might divide by zero T.T

    """
    v = tf.matmul(tf.linalg.inv(1.0*tf.identity(4)
    	-tf.scalar_mul(gamma,tf.transpose(T))),
    tf.matmul(tf.transpose(T),
    	r))
    	"""
    A = tf.diag([1.0,1.0,1.0,1.0])-tf.multiply(tf.transpose(T),.9)
    v = tf.matmul(tf.linalg.inv(A),tf.matmul(tf.transpose(T),r))

    return v,T



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

#smoothly produces a one-hot vector to mask instead of indexing directly
def hackyIndexIntoA(last_a,last_b):
	return last_a*last_b*tf.constant([[1.0],[0.0],[0.0],[0.0]],tf.float32)+last_a*(1-last_b)*tf.constant([[0.0],[1.0],[0.0],[0.0]],tf.float32)+(1-last_a)*last_b*tf.constant([[0.0],[0.0],[1.0],[0.0]],tf.float32)+(1-last_a)*(1-last_b)*tf.constant([[0.0],[0.0],[0.0],[1.0]],tf.float32)

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
va,Ta = exactDiscountedValue(cc,cd,dc,dd,a,b)
vb,Tb = exactDiscountedValue(cc,cd,dc,dd,b,a)
#^ok, one shortcoming of this is that, once they get into a cooperative rhythm, they don't train the other params
#so it might be good to randomize the pair of actions every 100 rounds, or to change ua to a sum
# ua = va[indexIntoA(memory[last_a],memory[last_b])]
# ub = vb[indexIntoA(memory[last_b],memory[last_a])]
#ok, replace indexing, which is generically nondifferentiable, with a dot product
ua = tf.matmul(tf.transpose(va),hackyIndexIntoA(last_a,last_b))
ub = tf.matmul(tf.transpose(vb),hackyIndexIntoA(last_b,last_a))



"""
ua = tf.reduce_sum(va,0)
ub = tf.reduce_sum(vb,0)
#uh, this is really sketchy though since they're not actually playing... lol, wtv
"""
#yeah, ua and ub should weight v by the expected time spent in each state
"""
state_dist_a = tf.linalg.inv((tf.diag([1.0,1.0,1.0,1.0])-tf.multiply(tf.transpose(Ta),.9)))
state_dist_a = state_dist_a[:,indexIntoA(memory[last_a],memory[last_b])]
state_dist_b = tf.linalg.inv((tf.diag([1.0,1.0,1.0,1.0])-tf.multiply(tf.transpose(Tb),.9)))
state_dist_b = state_dist_b[:,indexIntoA(memory[last_b],memory[last_a])]
ua = tf.reduce_sum(tf.multiply(va, state_dist_a))
ub = tf.reduce_sum(tf.multiply(vb, state_dist_b))
"""


#anyway, to compute the OpenAI LOLA agent, -- oh, we might Already be doing that depending on how
#GradientDescentOptimizer works, hm 

# The Gradient Descent Optimizer does the heavy lifting
#I take it that dua is the updated set of parameters for player A
dua = tf.train.GradientDescentOptimizer(0.001).minimize(0-ua, var_list=[a])
dub = tf.train.GradientDescentOptimizer(0.001).minimize(0-ub, var_list=[b])


#experiments:
#The main difficulty here is that there are too many dimensions to graph
#(i.e. each player's strategy is four dimensional.) 
#One way to reduce the dimensionality is to initialize each conditional probability P(c|xy) to be the same.
#We could also decrease the dimensionality by considering symmetric 2D strategies e.g. P(c|cx) and P(c|dx)
#Another thing I'm interested in is finding equilibria. We could look at all randomized initializations,
#pick out any eq that emerge, and then say things about stable eqs or more common eqs.
#I suppose we could also try implementing the strategies from the paper showing IPD includes the ultimatum game
#Ok, so create a list of a's and b's (hm, to be less noisy, I might even want a matrix i.e. run lots of inits to TFT)

#ha, given how I still don't get updates and whatnot, it could be easier to write a new python program that
#calls this one with different strategy inits

def reInitA(a):
	a[0] = 0;





with tf.Session() as session:
    session.run(model)
    for i in range(100000):        
        a_value = session.run(a, feed_dict=memory)
        b_value = session.run(b, feed_dict=memory)
        va_value = session.run(va, feed_dict=memory)
        Ta_value = session.run(Ta, feed_dict=memory)
        #print T_value
        
        #print every 10 turns I guess
        if i % 1000 == 0:
            print a_value
            print b_value
            print "Last pair of actions was {x:.3f} and {y:.3f}".format(x=int(memory[last_a]),y=int(memory[last_b]))
            print va_value
            print Ta_value

        if (sigmoid(a_value[indexIntoA(memory[last_a],memory[last_b])].item()) > uniform(0.0,1.0)):
        	last_a_temp = 1.0
        else:
        	last_a_temp = 0.0
        if (sigmoid(b_value[indexIntoA(memory[last_b],memory[last_a])].item()) > uniform(0.0,1.0)):
        	memory[last_b] = 1.0
        else: 
        	memory[last_b] = 0.0
        memory[last_a] = last_a_temp #not sure if I'm addressing correctly >.<
        time.sleep(.001)

        session.run(dua, feed_dict=memory) #I think this updates a?
        session.run(dub, feed_dict=memory)


        #another test can be to see if the strategies vary greatly over time

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





