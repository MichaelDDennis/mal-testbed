from tf_node_wrappers import *
from typing import TypeVar, Generic

Observation_Node = TypeVar("Observation_Node", bound='TFNodeWrapper')
Action_Node = TypeVar("Action_Node", bound='TFNodeWrapper')
Model_Node = TypeVar("Model_Node", bound='TFNodeWrapper')


class AgentModelNodeGenerator(Generic[Observation_Node, Action_Node]):

    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_action(self, observation_node: Observation_Node) -> Action_Node:
        raise Exception('The action function needs to be overridden')


class ModelBasedAgentModelNodeGenerator(Generic[Observation_Node, Action_Node, Model_Node],
                                        AgentModelNodeGenerator[Observation_Node, Action_Node]):

    def __init__(self, model_in: Model_Node,
                 update_in: Callable[[Model_Node, Observation_Node], Model_Node],
                 predict_in: Callable[[Model_Node, Observation_Node], Action_Node]) -> None:
        self.model = model_in
        self.update = update_in
        self.predict = predict_in

    def get_action(self, observation_node: Observation_Node) -> Action_Node:
        self.model = self.update(self.model, observation_node)
        return self.predict(self.model, observation_node)

