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



class ModelBasedAgent(Agent):

    def __init__(self, initial_model):
        super().__init__()
        self._model = initial_model

    def get_action(self, observation):
        self.update(observation)
        return self.predict(observation)

    def _get_model(self):
        return self._model

    @abstractmethod
    def update(self, observation):
        raise Exception('The update function needs to be overridden')

    @abstractmethod
    def predict(self, observation):
        raise Exception('The predict function needs to be overridden')



class GradientDecentBasedAgent(ModelBasedAgent):
    def __init__(self, initial_model, predict, utility, params, get_state):
        super().__init__(initial_model)
        self._predict = predict
        self._update = tf.train.GradientDescentOptimizer(0.01).minimize(0 - utility, var_list=[params])
        self._params = params
        self.__get_state=get_state

    def update(self, observation):
        session.run(self._update, feed_dict=self._get_state(observation))

    def predict(self, observation):
        action = session.run(self._predict, feed_dict=self._get_state(observation)).item()
        me = session.run(self._params)
        return action, me

    def _get_state(self, observation):
        return self.__get_state(observation)



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
    state = initial_state

    for i in range(1000):

        action_b = agent_b.get_action(observation_function_b(state))
        action_a = agent_a.get_action(observation_function_a(state))
         
        state = dynamics(state, action_a, action_b)

        print_state(state)
        time.sleep(1)
        

# State Dynamics which do not depend on the last state, and give a unique state for every action pair
def action_pair_dynamics(_, last_action_a, last_action_b):
    return {'lasta': last_action_a[0], 'lastb': last_action_b[0], 'a': last_action_a[1], 'b': last_action_b[1]}


# Observation Function for Fully Observable EnvironRments
def transparent_observation_function(state):
    return state


# Observation Function for Fully Observable EnvironRments
def reflective_observation_function(state):
    res = {'lasta': state['lastb'], 'lastb': state['lasta'], 'a': state['b'], 'b': state['a']}
    return res


def getMixedUtilFun(mat, pa, pb):
    u=0
    for i in range(len(mat)):
        for j in range(len(mat)):
            u=u+tf.multiply(tf.multiply(pa[i], pb[j]), mat[i][j])
    return u


def make_state(observation):
    return {opp: observation['b'], last_a: observation['lasta'], last_b: observation['lastb']}


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

    # Setting up tensor flow before running the simulation
    model = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(model)
        modelA = {'me': a, 'opp': []}
        modelB = {'me': b, 'opp': []}
        simulate({'lasta': 1.0, 'lastb': 1.0, 'a': [0.0, 100000.0, 0.0], 'b': [100000.0, 0.0, 0.0]},
                 action_pair_dynamics,
                 transparent_observation_function, reflective_observation_function,
                 GradientDecentBasedAgent(modelA, probca, ua, a, make_state),
                 GradientDecentBasedAgent(modelB, probcb, ub, b, make_state))
s

if __name__ == "__main__":
    main()
