from abc import abstractmethod
from typing import TypeVar, Generic, Callable
import tensorflow as tf
import numpy as np

Observation = TypeVar("Observation")
Action = TypeVar("Action")


# The purpose of this class is to serve as an interface for the simulation function.  It probably will never need to
# be changed, but will probably be sub-classed many times.
class Agent(Generic[Observation, Action]):
    def __init__(self):
        pass

    @abstractmethod
    def get_action(self, observation: Observation) -> Action:
        raise Exception('The action function needs to be overridden')


# This class is simply for convenience, so you can make agents based on the more natural update/predict breakdown.
class ModelBasedAgent(Agent[Observation, Action]):

    def __init__(self) -> None:
        super().__init__()

    def get_action(self, observation: Observation) -> Action:
        self.update(observation)
        return self.predict(observation)

    @abstractmethod
    def update(self, observation: Observation) -> None:
        raise Exception('The update function needs to be overridden')

    @abstractmethod
    def predict(self, observation: Observation) -> Action:
        raise Exception('The predict function needs to be overridden')


# This class implements the above interfaces so it can be used in simulations.  It takes responsibility for building the
# update mechanism and running it at the appropriate times, but must be given the appropriate computation graphs
class GradientDescentBasedAgent(ModelBasedAgent[Observation, Action]):
    def __init__(self, get_session, predict_node, utility_node, params_vars, get_state):
        super().__init__()
        self._get_session = get_session
        self._predict_node = predict_node
        self._update = tf.train.GradientDescentOptimizer(0.01).minimize(0 - utility_node, var_list=[params_vars])
        self._params = params_vars
        self._get_state = get_state

    def update(self, observation_val: Observation) -> None:
        self._get_session().run(self._update, feed_dict=self._get_state(observation_val))

    def predict(self, observation_val: Observation) -> Action:
        action_val = self._get_session().run(self._predict_node, feed_dict=self._get_state(observation_val))
        return action_val


class ConstantStrategyAgent(ModelBasedAgent[Observation, Action]):
    def __init__(self, get_session, predict_node, get_state):
        super().__init__()
        self._get_session = get_session
        self._predict_node = predict_node
        self._get_state = get_state

    def update(self, observation_val: Observation) -> None:
        pass

    def predict(self, observation_val: Observation) -> Action:
        action_val = self._get_session().run(self._predict_node, feed_dict=self._get_state(observation_val))
        return action_val


Model = TypeVar("Model")


class ModelActionPair(Generic[Model, Action]):

    def __init__(self, model_in: Model, action_in: Action) -> None:
        self.model = model_in
        self.action = action_in

    def get_model(self) -> Model:
        return self.model

    def get_action(self) -> Action:
        return self.action


# This class is a wrapper around agents that makes them pair their models with whatever action they take.  This forces
# them to be transparent as the other agent can read their model from their observations.
class TransparentAgentDecorator(Generic[Observation, Action, Model],
                                Agent[Observation, ModelActionPair[Model, Action]]):
    def __init__(self, agent: Agent[Observation, Action], get_model: Callable[[], Model]) -> None:
        super().__init__()
        self._agent = agent
        self._get_model = get_model

    def get_action(self, observation: Observation) -> ModelActionPair[Model, Action]:
        return ModelActionPair(self._get_model(), self._agent.get_action(observation))


def sample(distribution):
    r = np.random.random()
    running_sum = 0
    for i in range(len(distribution)):
        running_sum = running_sum + distribution[i]
        if running_sum > r:
            return i


Distribution = TypeVar("Distribution")


class ActionDistributionPair(Generic[Action, Distribution]):

    def __init__(self, action_in: Action, distribution_in: Distribution) -> None:
        self.action = action_in
        self.distribution = distribution_in

    def get_action(self) -> Action:
        return self.action

    def get_distribution(self) -> Distribution:
        return self.distribution


class SamplingAgentDecorator(Generic[Observation, Distribution],
                             Agent[Observation, ActionDistributionPair[int, Distribution]]):
    def __init__(self, agent: Agent[Observation, Distribution]) -> None:
        super().__init__()
        self._agent = agent

    def get_action(self, observation: Observation) -> ActionDistributionPair[int, Distribution]:
        action_distribution = self._agent.get_action(observation)
        return ActionDistributionPair(sample(action_distribution), action_distribution)


class NameAgentDecorator(Agent[Observation, Action]):

    def __init__(self, agent: Agent[Observation, Action], name: str) -> None:
        super().__init__()
        self._name = name
        self._agent = agent

    def get_action(self, observation: Observation) -> Action:
        return self._agent.get_action(observation)

    def __str__(self) -> str:
        return self._name
