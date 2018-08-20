from abc import abstractmethod
from typing import TypeVar, Generic, Callable, Tuple, Any
import tensorflow as tf


# Right now TFNodeWrappers are being used as a marker for this level of abstraction.  They are objects that aggregate
# TF Nodes into pieces with some sort of semantic meaning.  The end goal being easier debugging and having methods that
# work at a higher level of abstraction.
#
# By convention TensorFlowWrapperNodes will be immutable.  The underlying information should never change.
class TFNodeWrapper:
    pass


class ScalarNode(TFNodeWrapper):

    def __init__(self):
        raise Exception('cannot instantiate abstract class')

    @abstractmethod
    def get_val(self):
        raise Exception('the get_val function needs to be overrriden')


Val_Type = TypeVar("Val_Type")
Observation_Val = TypeVar("Observation_Val")


class InputNode(Generic[Observation_Val, Val_Type], TFNodeWrapper):

    def __init__(self, load_val_in: Callable[[Observation_Val], Val_Type]) -> None:
        self.load_val = load_val_in
        self.tf_node = tf.placeholder(tf.float32)

    def load(self, observation: Observation_Val) -> Tuple[Any, Val_Type]:
        return self.tf_node, self.load_val(observation)

    def get_val(self):
        return self.tf_node


class ConstNode(TFNodeWrapper):

    def __init__(self, const):
        self.tf_node = const

    def get_val(self):
        return self.tf_node



class TriVal_VariableNode(TFNodeWrapper):
    #TODO refactor to remove get_session
    def __init__(self, start_val: Val_Type, name: str, get_session_in) -> None:
        self.tf_node = tf.Variable(start_val, name)
        self.get_session = get_session_in

    def get_val(self) -> Val_Type:
        return self.get_session().run(self.tf_node)

    def get_tensor_flow_node(self):
        return self.tf_node


Action_X_Node = TypeVar("Action_X_Node", bound='TFNodeWrapper')
Action_Y_Node = TypeVar("Action_Y_Node", bound='TFNodeWrapper')
Model_X_Node = TypeVar("Model_X_Node", bound='TFNodeWrapper')
Model_Y_Node = TypeVar("Model_Y_Node", bound='TFNodeWrapper')
State_Node = TypeVar("State_Node", bound='TFNodeWrapper')


class ActionPairObservationNode(Generic[Action_X_Node, Action_Y_Node], TFNodeWrapper):

    def __init__(self, action_me_node_in: Action_X_Node, action_opp_node_in: Action_Y_Node) -> None:
        self.action_me_node = action_me_node_in
        self.action_opp_node = action_opp_node_in

    def get_last_me_action_node(self) -> Action_X_Node:
        return self.action_me_node

    def get_last_opp_action_node(self) -> Action_Y_Node:
        return self.action_opp_node


class ActionPairStateNode(Generic[Action_X_Node, Action_Y_Node], TFNodeWrapper):

    def __init__(self, action_x_node_in: Action_X_Node, action_y_node_in: Action_Y_Node) -> None:
        self.action_x_node = action_x_node_in
        self.action_y_node = action_y_node_in

    def get_last_x_action_node(self):
        return self.action_x_node

    def get_last_y_action_node(self):
        return self.action_y_node


class TotalStateNode(Generic[Model_X_Node, Model_Y_Node, State_Node], TFNodeWrapper):

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
