import tensorflow as tf
import numpy as np
import time

last_a =tf.placeholder(tf.float32)
last_b =tf.placeholder(tf.float32)


opp    =tf.placeholder(tf.float32,(3,))


cc=400
cd=0
dc=401
dd=50



def getUtilFun(cc,cd,dc,dd,probca,probcb):
    u = tf.multiply(tf.multiply(probca,  probcb),cc)+tf.multiply(tf.multiply(probca, 1- probcb),cd)+tf.multiply(tf.multiply(1-probca,  probcb),dc)+tf.multiply(tf.multiply(1-probca, 1- probcb),dd)
    return u



#All of the variables and Update for A
a = tf.Variable([0.0,100000.0,0.0], name="a")

probca     =tf.multiply(tf.tanh(tf.multiply(  a[0],last_a)+tf.multiply(  a[1],last_b)+  a[2]),0.5)+0.5
probcb_opp =tf.multiply(tf.tanh(tf.multiply(opp[0],last_a)+tf.multiply(opp[1],last_b)+opp[2]),0.5)+0.5

probca2    =tf.multiply(tf.tanh(tf.multiply(  a[0],probca)+tf.multiply(  a[1],probcb_opp)+  a[2]),0.5)+0.5
probcb2_opp=tf.multiply(tf.tanh(tf.multiply(opp[0],probca)+tf.multiply(opp[1],probcb_opp)+opp[2]),0.5)+0.5




ua1= getUtilFun(cc,cd,dc,dd,probca,probcb_opp)
ua2= getUtilFun(cc,cd,dc,dd,probca2,probcb2_opp)
ua=ua1+ua2
dua = tf.train.GradientDescentOptimizer(0.01).minimize(0-ua, var_list=[a])


#All of the Variables and Update for B
b = tf.Variable([100000.0,0.0,0.0], name="b")

probca_opp  =tf.multiply(tf.tanh(tf.multiply(opp[0],last_a)+tf.multiply(opp[1],last_b)+opp[2]),0.5)+0.5
probcb      =tf.multiply(tf.tanh(tf.multiply(  b[0],last_a)+tf.multiply(  b[1],last_b)+  b[2]),0.5)+0.5

probca2_opp =tf.multiply(tf.tanh(tf.multiply(opp[0],probca_opp)+tf.multiply(opp[1],probcb)+opp[2]),0.5)+0.
probcb2     =tf.multiply(tf.tanh(tf.multiply(  b[0],probca_opp)+tf.multiply(  b[1],probcb)+  b[2]),0.5)+0.5



ub1= getUtilFun(cc,cd,dc,dd,probcb,probca_opp)
ub2= getUtilFun(cc,cd,dc,dd,probcb2,probca2_opp)
ub=ub1+ub2
dub = tf.train.GradientDescentOptimizer(0.01).minimize(0-ub, var_list=[b])


#The Default Print Function
def printF(s):
    print(s)

# Simulate is the Main function we will be studying, the structure here is meant to force isolation between 
# different parts of the code base so it is easier to tell what is going on and who knows about who.  Ideally
# there would be no global state, and all model parameters, would be fed through ma/mb.
#
# One specific way in which this helps avoid confusion is in the seperating the game itself from how they 
# update their models.  Though both will have similar structure, one will be in the agents head and the 
# other will be in the real world.  Only the first needs to be differentiable and they need not always match.
#
# For clarity, the types of all of the arguments are as follows:
#
# Initial State: State
# Dyn: State x ActA x ActB --> State
# ObsA: State -> ObsA
# ObsB: State -> ObsB
#
# A: (InitialMA,DTA)
#   InitialMA: HiddenStateA
#   DTA: ObsA x HiddenStateA --> ActA x HiddenStateA
#
# B: (InitialMB,DTB)
#   InitialMB: HiddenStateB
#   DTB: ObsB x HiddenStateB --> ActB x HiddenStateB
#
def Simulate(InitialState,Dyn,ObsA,ObsB,transparencyA2B,transparencyB2A,A,B,printState=printF):
    # Normal TensorFlow - initialize values, create a session and run the model
     
    state=InitialState
    ma=A[0]
    mb=B[0]
    DTA=A[1]
    DTB=B[1]

    for i in range(1000):
        transparencyA2B(ma) 
        b,mb=DTB(ObsB(state),mb)
        
        transparencyB2A(mb)
        a,ma=DTA(ObsA(state),ma)
         
        state=Dyn(state,a,b)
        
        printState(state)
        time.sleep(1)
        


#State Dynamics which do not depend on the last state, and give a unique state for every action pair
def ActionPairDyn(s,a,b):
    return {'lasta':a,'lastb':b}

#Observation Function for Fully Observable Enviroments
def TransparentObs(s):
    return s

#Some Dummy Decsion Rules for testing purposes
def TitForTatA(o,m):
    return (o['lastb'],m)

def TitForTatB(o,m):
    return (o['lasta'],m)





def ModelBasedAgent(initalModel,update,predict):
    def ModelBasedDT(o,m):
        newM=update(m,o)
        return (predict(newM,o),newM)
    return (initalModel,ModelBasedDT)

def GradDecentUpdate(m,o):
    state={opp:m['opp'],last_a:o['lasta'],last_b:o['lastb']}
    session.run(m['update'], feed_dict=state)
    return m

def GradDecentPredict(m,o):
    state={opp:m['opp'],last_a:o['lasta'],last_b:o['lastb']}
    return session.run(m['predict'], feed_dict=state).item()

def Transparency(a):
    def temp(mb):
        a['opp']=session.run(mb['me'])
    return temp



#Setting up tensor flow before running the simulation
model = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(model)
    modelA={'me':a,'opp':[] , 'update':dua,'predict':probca}
    modelB={'me':b,'opp':[], 'update':dub,'predict':probcb}
    Simulate({'lasta':1.0,'lastb':1.0},ActionPairDyn,TransparentObs,TransparentObs,Transparency(modelB),Transparency(modelA),ModelBasedAgent(modelA,GradDecentUpdate,GradDecentPredict),ModelBasedAgent(modelB,GradDecentUpdate,GradDecentPredict))






