from abc import ABC, abstractmethod
from numpy import random
import tensorflow as tf


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
    # The following line is so pycharm knows that it not knowing where choice is is numpy's fault and not mine.
    # noinspection PyUnresolvedReferences
    return (random.choice(range(len(distribution)), 1, distribution)[0]*1.0).item()


class SamplingAgentDecorator(Agent):
    def __init__(self, agent):
        super().__init__()
        self._agent = agent

    def get_action(self, observation):
        action = self._agent.get_action(observation)
        return sample(action)

class NameAgentDecorator(Agent):

    def __init__(self,agent, name):
        self._name=name
        self._agent=agent

    def get_action(self, observation):
        return self._agent.get_action(observation)

    def __str__(self):
        return self._name