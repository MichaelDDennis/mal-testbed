from abc import abstractmethod
from typing import TypeVar, Generic, Callable, Tuple, Any
import tensorflow as tf

Val_Type = TypeVar("Val_Type")
Observation_Val = TypeVar("Observation_Val")

# By convention TensorFlowWrapperNodes will be immutable.  The underlying node should never change.
class TensorFlowWrapperNode():


    # Get the children tensor flow node wrappers for pretty print
    def get_children(self):
        pass


    # Make sure they make string work
    def ___str__(self):
        pass


    # Gives a function that will get the current values of this node given a session
    def get_current_value_function(self):
        pass


    # Get the tensor flow node that this is wrapping
    def get_tensor_flow_node(self):
        pass


    # Gets a list of tensor flow input nodes on which this nodes computation depends.
    def get_placeholder_dependencies(self):
        pass


class InputNode(Generic[Observation_Val, Val_Type]):

    def __init__(self, load_val_in: Callable[[Observation_Val], Val_Type]) -> None:
        self.load_val = load_val_in
        self.tf_node = tf.placeholder(tf.float32)

    def load(self, observation: Observation_Val) -> Tuple[Any, Val_Type]:
        return self.tf_node, self.load_val(observation)


class ConstNode:

    def __init__(self, const):
        self.tf_node = const


class VariableNode(Generic[Val_Type]):
    #TODO refactor to remove get_session
    def __init__(self, start_val: Val_Type, name: str, get_session_in) -> None:
        self.tf_node = tf.Variable(start_val, name)
        self.get_session = get_session_in

    def get_val(self) -> Val_Type:
        return self.get_session().run(self.tf_node)

Observation_Node = TypeVar("Observation_Node")
Action_Node = TypeVar("Action_Node")
Model_Node = TypeVar("Model_Node")


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


Action_X_Node = TypeVar("Action_X_Node")
Action_Y_Node = TypeVar("Action_Y_Node")


class ActionPairObservationNode(Generic[Action_X_Node, Action_Y_Node]):

    def __init__(self, action_me_node_in: Action_X_Node, action_opp_node_in: Action_Y_Node) -> None:
        self.action_me_node = action_me_node_in
        self.action_opp_node = action_opp_node_in

    def get_last_me_action_node(self):
        return self.action_me_node

    def get_last_opp_action_node(self):
        return self.action_opp_node


class ActionPairStateNode(Generic[Action_X_Node, Action_Y_Node]):

    def __init__(self, action_x_node_in: Action_X_Node, action_y_node_in: Action_Y_Node) -> None:
        self.action_x_node = action_x_node_in
        self.action_y_node = action_y_node_in

    def get_last_x_action_node(self):
        return self.action_x_node

    def get_last_y_action_node(self):
        return self.action_y_node


Model_X_Node = TypeVar("Model_X_Node")
Model_Y_Node = TypeVar("Model_Y_Node")
State_Node = TypeVar("State_Node")


class TotalStateNode(Generic[Model_X_Node, Model_Y_Node, State_Node]):
    # include probability and depth

    def __init__(self, state_in, me_params_node_in,  opp_params_node_in,  depth_in,  prob_in):
        self.state = state_in
        self.me_params_node = me_params_node_in
        self.opp_params_node = opp_params_node_in
        self.depth = depth_in
        self.prob = prob_in

    def get_state(self):
        return self.state

    def get_me_params_node(self):
        return self.me_params_node

    def get_opp_params_node(self):
        return self.opp_params_node

    def get_depth(self):
        return self.depth

    def get_prob(self):
        return self.prob
