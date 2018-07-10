import tensorflow as tf
from abc import ABC, abstractmethod
import time


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
        self.__get_state = get_state

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


def get_mixed_utility_function(mat, pa, pb):
    u = 0
    for i in range(len(mat)):
        for j in range(len(mat)):
            u = u+tf.multiply(tf.multiply(pa[i], pb[j]), mat[i][j])
    return u


def get_discounted_utility(payoff, probabiltiy_pair_ist, discount=1):
    res = 0
    current_value = 1
    for probA, probB in probabiltiy_pair_ist:
        res += get_mixed_utility_function(payoff, probA, probB)*current_value
        current_value *= discount
    return res


def make_agent(start_vector):
    last_me = tf.placeholder(tf.float32)
    last_opp = tf.placeholder(tf.float32)
    opp = tf.placeholder(tf.float32, (3,))
    me = tf.Variable(start_vector, name="me")

    payoff = [[400, 0],
              [401, 50]]

    def me_model(last_me_action, last_opp_action, me_vars):
        return tf.multiply(tf.tanh(tf.multiply(me_vars[0], last_me_action) +
                                   tf.multiply(me_vars[1], last_opp_action) + me_vars[2]), 0.5) + 0.5

    def opp_model(last_me_action, last_opp_action, opp_vars):
        return tf.multiply(tf.tanh(tf.multiply(opp_vars[0], last_opp_action) +
                                   tf.multiply(opp_vars[1], last_me_action) + opp_vars[2]), 0.5) + 0.5

    probcme = me_model(last_me, last_opp, me)
    probcopp = opp_model(last_me, last_opp, opp)

    probcme2 = me_model(probcme, probcopp, me)
    probcopp2 = opp_model(probcme, probcopp, me)

    probcmePA = [probcme, 1 - probcme]
    probcoppPA = [probcopp, 1 - probcopp]
    probcme2PA = [probcme2, 1 - probcme2]
    probcopp2PA = [probcopp2, 1 - probcopp2]

    u = get_discounted_utility(payoff, [(probcmePA, probcoppPA), (probcme2PA, probcopp2PA)])

    modelA = {'me': me}

    def make_state(observation):
        return {opp: observation['b'], last_me: observation['lasta'], last_opp: observation['lastb']}

    return GradientDecentBasedAgent(modelA, probcme, u, me, make_state)


def main():
    global session

    agent_a = make_agent([0.0, 100000.0, 0.0])
    agent_b = make_agent([100000.0, 0.0, 0.0])

    # Setting up tensor flow before running the simulation
    model = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(model)

        initial_state = {'lasta': 1.0, 'lastb': 1.0, 'a': [0.0, 100000.0, 0.0], 'b': [100000.0, 0.0, 0.0]}
        simulate(initial_state, action_pair_dynamics, transparent_observation_function, reflective_observation_function,
                 agent_a, agent_b)


if __name__ == "__main__":
    main()
