import tensorflow as tf
from abc import ABC, abstractmethod
import time

# refactor to remove these TODO
session=[]
last_a,last_b=0,0
opp=0


class Agent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, observation):
        raise Exception('The action function needs to be overridden')

    @abstractmethod
    def load_opp_model(self,opp):
        raise Exception('The load_opp_model function needs to be overridden')


class ModelBasedAgent(Agent):

    def __init__(self, initial_model, update, predict):
        super().__init__()
        self._model = initial_model
        self._update = update
        self._predict = predict

    def get_action(self, observation):
        self._model = self._update(self._model, observation)
        return self._predict(self._model, observation)

    def load_opp_model(self, opp):
        self._model['opp'] = session.run(opp._model['me'])


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
def simulate(initial_state, dynamics, observation_function_a, observation_function_b,
             agent_a: Agent, agent_b: Agent, print_state=print):
    # Normal TensorFlow - initialize values, create a session and run the model
    agent_b.load_opp_model(agent_a)
    agent_a.load_opp_model(agent_b)
    state = initial_state

    for i in range(1000):

        action_b = agent_b.get_action(observation_function_b(state))
        action_a = agent_a.get_action(observation_function_a(state))
         
        state = dynamics(state, action_a, action_b)

        agent_b.load_opp_model(agent_b)
        agent_b.load_opp_model(agent_a)

        print_state(state)
        time.sleep(1)
        

# State Dynamics which do not depend on the last state, and give a unique state for every action pair
def action_pair_dynamics(_, last_action_a, last_action_b):
    return {'lasta': last_action_a, 'lastb': last_action_b}


# Observation Function for Fully Observable EnvironRments
def transparent_observation_function(state):
    return state


def gradient_decent_update(memory, observation):
    state = {opp: memory['opp'], last_a: observation['lasta'], last_b: observation['lastb']}
    session.run(memory['update'], feed_dict=state)
    return memory


def gradient_decent_predict(m, o):
    state = {opp: m['opp'], last_a: o['lasta'], last_b: o['lastb']}
    return session.run(m['predict'], feed_dict=state).item()


def getMixedUtilFun(mat, pa, pb):
    u=0
    for i in range(len(mat)):
        for j in range(len(mat)):
            u=u+tf.multiply(tf.multiply(pa[i], pb[j]), mat[i][j])
    return u


def main():
    global session, opp, last_a, last_b

    last_a = tf.placeholder(tf.float32)
    last_b = tf.placeholder(tf.float32)
    opp = tf.placeholder(tf.float32, (3,))

    payoff = [[400, 0],
              [401, 50]]

    # All of the variables and Update for A
    a = tf.Variable([0.0, 100000.0, 0.0], name="a")

    probca = tf.multiply(tf.tanh(tf.multiply(a[0], last_a) + tf.multiply(a[1], last_b) + a[2]), 0.5) + 0.5
    probcb_opp = tf.multiply(tf.tanh(tf.multiply(opp[0], last_a) + tf.multiply(opp[1], last_b) + opp[2]), 0.5) + 0.5

    probca2 = tf.multiply(tf.tanh(tf.multiply(a[0], probca) + tf.multiply(a[1], probcb_opp) + a[2]), 0.5) + 0.5
    probcb2_opp = tf.multiply(tf.tanh(tf.multiply(opp[0], probca) + tf.multiply(opp[1], probcb_opp) + opp[2]),
                              0.5) + 0.5

    probcaPA = [probca, 1 - probca]
    probcb_oppPA = [probcb_opp, 1 - probcb_opp]
    probca2PA = [probca2, 1 - probca2]
    probcb2_oppPA = [probcb2_opp, 1 - probcb2_opp]

    ua1 = getMixedUtilFun(payoff, probcaPA, probcb_oppPA)
    ua2 = getMixedUtilFun(payoff, probca2PA, probcb2_oppPA)
    ua = ua1 + ua2
    dua = tf.train.GradientDescentOptimizer(0.01).minimize(0 - ua, var_list=[a])

    # All of the Variables and Update for B
    b = tf.Variable([100000.0, 0.0, 0.0], name="b")

    probca_opp = tf.multiply(tf.tanh(tf.multiply(opp[0], last_a) + tf.multiply(opp[1], last_b) + opp[2]), 0.5) + 0.5
    probcb = tf.multiply(tf.tanh(tf.multiply(b[0], last_a) + tf.multiply(b[1], last_b) + b[2]), 0.5) + 0.5

    probca2_opp = tf.multiply(tf.tanh(tf.multiply(opp[0], probca_opp) + tf.multiply(opp[1], probcb) + opp[2]),
                              0.5) + 0.5
    probcb2 = tf.multiply(tf.tanh(tf.multiply(b[0], probca_opp) + tf.multiply(b[1], probcb) + b[2]), 0.5) + 0.5

    probca_oppPA = [probca_opp, 1 - probca_opp]
    probcbPA = [probcb, 1 - probcb]
    probca2_oppPA = [probca2_opp, 1 - probca2_opp]
    probcb2PA = [probcb2, 1 - probcb2]

    ub1 = getMixedUtilFun(payoff, probcbPA, probca_oppPA)
    ub2 = getMixedUtilFun(payoff, probcb2PA, probca2_oppPA)
    ub = ub1 + ub2
    dub = tf.train.GradientDescentOptimizer(0.01).minimize(0 - ub, var_list=[b])

    # Setting up tensor flow before running the simulation
    model = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(model)
        modelA = {'me': a, 'opp': [], 'update': dua, 'predict': probca}
        modelB = {'me': b, 'opp': [], 'update': dub, 'predict': probcb}
        simulate({'lasta': 1.0, 'lastb': 1.0}, action_pair_dynamics,
                 transparent_observation_function, transparent_observation_function,
                 ModelBasedAgent(modelA, gradient_decent_update, gradient_decent_predict),
                 ModelBasedAgent(modelB, gradient_decent_update, gradient_decent_predict))


if __name__ == "__main__":
    main()
