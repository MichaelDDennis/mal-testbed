import tensorflow as tf
from abc import ABC, abstractmethod
import time


# TODO Break everything out into separate files

# The purpose of this class is to serve as an interface for the simulation function.  It probably will never need to
# be changed, but will probably be sub-classed many times.
class Agent(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, observation):
        raise Exception('The action function needs to be overridden')


# This class is simply for convenience, so you can make agents based on the more natural update/predict breakdown.
class ModelBasedAgent(Agent):

    def __init__(self):
        super().__init__()

    def get_action(self, observation):
        self.update(observation)
        return self.predict(observation)

    @abstractmethod
    def update(self, observation):
        raise Exception('The update function needs to be overridden')

    @abstractmethod
    def predict(self, observation):
        raise Exception('The predict function needs to be overridden')


# This class implements the above interfaces so it can be used in simulations.  It takes responsibility for building the
# update mechanism and running it at the appropriate times, but must be given the appropriate computation graphs
class GradientDecentBasedAgent(ModelBasedAgent):
    def __init__(self, predict, utility, params, get_state):
        super().__init__()
        self._predict = predict
        self._update = tf.train.GradientDescentOptimizer(0.01).minimize(0 - utility, var_list=[params])
        self._params = params
        self.__get_state = get_state

    def update(self, observation):
        session.run(self._update, feed_dict=self._get_state(observation))

    def predict(self, observation):
        action = session.run(self._predict, feed_dict=self._get_state(observation)).item()
        return action

    def _get_state(self, observation):
        return self.__get_state(observation)


# This class is a wrapper around agents that makes them pair their models with whatever action they take.  This forces
# them to be transparent as the other agent can read their model from their observations.
class TransparentAgentDecorator(Agent):
    def __init__(self, agent, get_model):
        super().__init__()
        self._agent = agent
        self._get_model = get_model

    def get_action(self, observation):
        return {'action': self._agent.get_action(observation), 'model': self._get_model()}


# Simulate is the Main function we will be studying, the structure here is meant to force isolation between 
# different parts of the code base so it is easier to tell what is going on and who knows about who.
#
# One specific way in which this helps avoid confusion is in the separating the game itself from how they
# update their models.  Though both will have similar structure, one will be in the agents head and the 
# other will be in the real world.  Only the first needs to be differentiable and they need not always match.
#
# I also made this a streaming iterator so that interaction with it is easier and analysis of results can be seen
# as soon as possible.
#
# For clarity, the types of all of the arguments are as follows:
#
# Initial State: State
# Dyn: State x ActA x ActB --> State
# ObsA: State -> ObsA
# ObsB: State -> ObsB
#
# A: Agent
# B: Agent
#
def simulate(initial_state, dynamics, observation_function_a, observation_function_b,
             agent_a: Agent, agent_b: Agent):
    # Normal TensorFlow - initialize values, create a session and run the model
    state = initial_state

    for i in range(1000):

        action_b = agent_b.get_action(observation_function_b(state))
        action_a = agent_a.get_action(observation_function_a(state))
         
        state = dynamics(state, action_a, action_b)

        yield state


# Simulation Building Tools


# State Dynamics which do not depend on the last state, and give a unique state for every action pair
def action_pair_dynamics(_, last_action_a, last_action_b):
    return {'last_action_a': last_action_a, 'last_action_b': last_action_b}


# Observation Function for Fully Observable Environments
def full_observation_function(state):
    return state


# Reflects the observation so that the game is symmetric
def reflective_pair_observation_function(state):
    res = {'last_action_a': state['last_action_b'], 'last_action_b': state['last_action_a']}
    return res


# Stream Processing Tools


# This simply iterates through the states as they are produced from the simulation, prints them, and hands them to
# another stream
def print_state_decorator(simulation):
    for s in simulation:
        print(s)
        yield s


# This simply slows down the stream
def slow_sim_decorator(simulation, delay):
    for s in simulation:
        yield s
        time.sleep(delay)


# This runs the stream until the end
def run_sim(simulation):
    for _ in simulation:
        pass


# Computation Graph Building Tools


# This produces a utility function, from probability distributions to
def get_mixed_utility_function(mat):
    def utility_function(state):

        a = state['me_action_node']
        b = state['opp_action_node']
        pa = [a, 1 - a]
        pb = [b, 1 - b]

        u = 0
        for i in range(len(mat)):
            for j in range(len(mat)):
                u = u+tf.multiply(tf.multiply(pa[i], pb[j]), mat[i][j])
        return u
    return utility_function


def get_discounted_utility(reward, state_list, discount=1):
    res = 0
    current_value = 1
    for state in state_list:
        res += reward(state) * current_value
        current_value *= discount
    return res


def make_agent(start_vector):
    last_me = tf.placeholder(tf.float32)
    last_opp = tf.placeholder(tf.float32)
    opp = tf.placeholder(tf.float32, (3,))
    me = tf.Variable(start_vector, name="me")

    payoff = [[400, 0],
              [401, 50]]

    # TODO add easy way to bound probabilities within the computation graphs
    # TODO add an easy way to bound probabilities during updates
    # TODO add an easy way to build "complete" policy spaces
    def me_model(observation, me_vars):
        return tf.multiply(tf.tanh(tf.multiply(me_vars[0], observation['me_action_node']) +
                                   tf.multiply(me_vars[1], observation['opp_action_node']) + me_vars[2]), 0.5) + 0.5

    def opp_model(observation, opp_vars):
        return tf.multiply(tf.tanh(tf.multiply(opp_vars[0], observation['opp_action_node']) +
                                   tf.multiply(opp_vars[1], observation['me_action_node']) + opp_vars[2]), 0.5) + 0.5

    # TODO Fix Dynamics Model so that it is sampling-based instead of distribution game.
    def dyn_model(_, me_action, them_action):
        return {'me_action_node': me_action, 'opp_action_node': them_action}

    def me_observation_model(state):
        return state

    def opp_observation_model(state):
        return {'me_action_node': state['opp_action_node'], 'opp_action_node': state['me_action_node']}

    # TODO add opponent update models that are more realistic (gradient decent based)
    def me_update_model(me, _):
        return me

    def opp_update_model(opp, _):
        return opp

    opp_cur = opp
    me_cur = me
    initial_state = {'me_action_node': last_me, 'opp_action_node': last_opp}
    state = initial_state
    states=[]
    for i in range(1):
        state = dyn_model(state, me_model(me_observation_model(state), me_cur),
                          opp_model(opp_observation_model(state), opp_cur))
        me_cur = me_update_model(me, me_observation_model(state))
        opp_cur = opp_update_model(opp, opp_observation_model(state))
        states.append(state)

    u = get_discounted_utility(get_mixed_utility_function(payoff), states)

    # TODO Make actions and observations into objects so you don't have to keep passing around hash maps
    # TODO add type checking
    def make_state(observation):
        return {opp: observation['last_action_a']['model'], last_me: observation['last_action_a']['action'],
                last_opp: observation['last_action_b']['action']}

    def get_model():
        return session.run(me)

    return TransparentAgentDecorator(GradientDecentBasedAgent(me_model(me_observation_model(initial_state), me),
                                                              u, me, make_state), get_model)


def main():
    global session

    agent_a = make_agent([0.0, 100000.0, 0.0])
    agent_b = make_agent([100000.0, 0.0, 0.0])

    # Setting up tensor flow before running the simulation
    model = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(model)

        initial_state = {'last_action_a': {'action': 1.0, 'model': [0.0, 100000.0, 0.0]},
                         'last_action_b': {'action': 1.0, 'model': [100000.0, 0.0, 0.0]}}
        simulation = simulate(initial_state, action_pair_dynamics, full_observation_function,
                              reflective_pair_observation_function, agent_a, agent_b)

        run_sim(print_state_decorator(slow_sim_decorator(simulation, 1)))


if __name__ == "__main__":
    main()
