from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np


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
class GradientDescentBasedAgent(ModelBasedAgent):
    def __init__(self, get_session, predict_node, utility_node, params_vars, get_state):
        super().__init__()
        self._get_session = get_session
        self._predict_node = predict_node
        self._update = tf.train.GradientDescentOptimizer(0.01).minimize(0 - utility_node, var_list=[params_vars])
        self._params = params_vars
        self._get_state = get_state

    def update(self, observation_val):
        self._get_session().run(self._update, feed_dict=self._get_state(observation_val))

    def predict(self, observation_val):
        action_val = self._get_session().run(self._predict_node, feed_dict=self._get_state(observation_val))
        return action_val

class ConstantStrategyAgent(ModelBasedAgent):
    #TODO: get rid of the GradientDescentOptimizer lol
    def __init__(self, get_session, predict_node, get_state):
        super().__init__()
        self._get_session = get_session
        self._predict_node = predict_node
        self._get_state = get_state

    def update(self, observation_val):
        pass

    def predict(self, observation_val):
        action_val = self._get_session().run(self._predict_node, feed_dict=self._get_state(observation_val))
        return action_val


# This class is a wrapper around agents that makes them pair their models with whatever action they take.  This forces
# them to be transparent as the other agent can read their model from their observations.
class TransparentAgentDecorator(Agent):
    def __init__(self, agent, get_model):
        super().__init__()
        self._agent = agent
        self._get_model = get_model

    def get_action(self, observation):
        return {'action': self._agent.get_action(observation), 'model': self._get_model()}


def sample(distribution):
    r = np.random.random()
    running_sum = 0
    for i in range(len(distribution)):
        running_sum = running_sum + distribution[i]
        if running_sum > r:
            return i
    return len(distribution)-1


class SamplingAgentDecorator(Agent):
    def __init__(self, agent):
        super().__init__()
        self._agent = agent

    def get_action(self, observation):
        action_distribution = self._agent.get_action(observation)
        return {"sample": sample(action_distribution), "distribution": action_distribution}


class NameAgentDecorator(Agent):

    def __init__(self, agent, name):
        super().__init__()
        self._name = name
        self._agent = agent

    def get_action(self, observation):
        return self._agent.get_action(observation)

    def __str__(self):
        return self._name
