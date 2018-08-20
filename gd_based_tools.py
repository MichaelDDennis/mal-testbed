from agents import *
from simulation import *
import tensorflow as tf
from typing import Any, TypeVar, Tuple, Mapping
from tf_node_wrapper_generators import *


def load_me_model(observation: ActionPairObservation[ModelActionPair, Any]):
    return observation.get_last_me_action().get_model()


def load_opp_model(observation: ActionPairObservation[Any, ModelActionPair]):
    return observation.get_last_opp_action().get_model()


def load_me_action(observation: ActionPairObservation[ModelActionPair[Any, ActionDistributionPair], Any]):
    return observation.get_last_me_action().get_action().get_action()


def load_opp_action(observation: ActionPairObservation[Any, ModelActionPair[Any, ActionDistributionPair]]):
    return observation.get_last_opp_action().get_action().get_action()


def basic_3_val_predict(me_vars: TriVal_VariableNode, observation: ActionPairObservationNode[ScalarNode, ScalarNode]):
    prob_d = bound_probabilities(tf.multiply(me_vars.tf_node[0], observation.get_last_me_action_node().get_val() - 0.5) +
                                 tf.multiply(me_vars.tf_node[1], observation.get_last_opp_action_node().get_val() - 0.5) + me_vars.tf_node[2])
    return [1 - prob_d, prob_d]


def empty_update(me_vars, _):
    return me_vars


def load_inputs(input_nodes: List[InputNode[Observation, Val_Type]]) -> Callable[[Observation], Mapping]:
    def load(observation: Observation) -> Mapping:
        result = {}
        for input_node in input_nodes:
            node, val = input_node.load(observation)
            result[node] = val
        return result
    return load


Action_Me_Val = TypeVar("Action_Me_Val")
Observation_Me_Val = TypeVar("Observation_Me_Val")
Model_Me_Val = TypeVar("Model_Me_Val")
Action_Opp_Val = TypeVar("Action_Opp_Val")
Observation_Opp_Val = TypeVar("Observation_Opp_Val")
Model_Opp_Val = TypeVar("Model_Opp_Val")


def initial_state_maker(me_action: Action_Me_Val, opp_action: Action_Opp_Val,
                        me_model: Model_Me_Val, opp_model: Model_Opp_Val)\
                -> ActionPairState[ModelActionPair[Model_Me_Val, ActionDistributionPair[Action_Me_Val]],
                                   ModelActionPair[Model_Opp_Val, ActionDistributionPair[Action_Opp_Val]]]:
    return ActionPairState(ModelActionPair(me_model, ActionDistributionPair(me_action, [])),
                           ModelActionPair(opp_model, ActionDistributionPair(opp_action, [])))


def make_constant_agent(get_session, start_vector, name):
    last_me_action_node = InputNode(load_me_action)
    last_opp_action_node = InputNode(load_opp_action)
    inputs = [last_me_action_node, last_opp_action_node]
    me_node = TriVal_VariableNode(start_vector, "me", get_session)

    constant_agent_model = ModelBasedAgentModelNodeGenerator(me_node, empty_update, basic_3_val_predict)

    def get_model():
        return start_vector

    return TransparentAgentDecorator(SamplingAgentDecorator(NameAgentDecorator(ConstantStrategyAgent(
        get_session,
        constant_agent_model.get_action(ActionPairObservationNode(last_me_action_node, last_opp_action_node)),
        load_inputs(inputs)), name)), get_model)








# This produces a utility function, from probability distributions to
# note that we're not actually using this now
def get_mixed_utility_function(mat):
    def utility_function(state):

        a = state['me_action_node']
        b = state['opp_action_node']
        pa = [1-a, a]
        pb = [1-b, b]

        u = 0.0
        for i in range(len(mat)):
            for j in range(len(mat)):
                u = u+tf.multiply(tf.multiply(pa[i], pb[j]), mat[i][j])
        return u
    return utility_function


# This produces a utility function, from probability distributions to
def get_utility_function_from_payoff(mat):
    def utility_function(state):

        a = state.get_last_x_action_node().tf_node
        b = state.get_last_y_action_node().tf_node

        return mat[a][b]
    return utility_function


def get_utility_of_states(reward, state_list):
    res = 0.0
    for state, prob in state_list:
        r = tf.multiply(reward(state), prob)
        res += r
    return res


def bound_probabilities(input_node):
    return tf.multiply(tf.tanh(input_node), 0.5)+0.5


def get_utility_node(reward_model, dyn_model, me_model, opp_model,
                     me_observation_model, opp_observation_model, me_update_model, opp_update_model,
                     initial_state_to_process, max_depth, markov_sim = False):

    states = []
    full_states_to_process = [initial_state_to_process]
    #TODO: prevent exponential blowup by taking advantage of Markov property (although this won't work with learning...)
    #TODO: also prevent exponential blowup in learning rate by renormalizing
    #TODO: Replace this with the exact calculation that involved the matrix inverse (this involves finding the transition matrix)
    for full_state in full_states_to_process:
        state = full_state.state
        me_cur = full_state.me_params_node
        opp_cur = full_state.opp_params_node
        action_distr_me = me_model(me_observation_model(state), me_cur)
        action_distr_opp = opp_model(opp_observation_model(state), opp_cur)

        for action_me in range(2):
            for action_opp in range(2):
                new_state = dyn_model(state, ConstNode(action_me), ConstNode(action_opp))
                prob = tf.multiply(action_distr_me[action_me], action_distr_opp[action_opp])
                states.append((new_state, tf.multiply(full_state.get_prob(), prob)))

                if full_state.get_depth() < max_depth:
                    full_states_to_process.append(TotalStateNode(new_state,
                                                                 me_update_model(me_cur, me_observation_model(state)),
                                                                 opp_update_model(opp_cur,
                                                                                  opp_observation_model(state)),
                                                                 full_state.get_depth()+1,
                                                                 tf.multiply(full_state.get_prob(), prob)))

    return get_utility_of_states(reward_model, states)

#this calculates discounted future utility assuming an MDP with a fixed, memory-1 transition matrix,
#this transition matrix will then be acted on in a single step of learning
def get_exact_discounted_utility_node(payoff_matrix, dyn_model, me_model,opp_model,me_observation_model, opp_observation_model,initial_state_to_process):
    #construct transition matrix
    # returns vector v with v_xy the value of that state for A (e.g. v_cd is the value of that state for A after both players cooperated)
    # so to get values for b swap cd with dc and a with b
    # if we wanted no discounting, then we would solve for the stationary vector
    # see "Iterated Prisoner's Dilemma contains strategies that dominate any evolutionary opponent"
    #TODO: maybe refactor for arbitrary matrices (certainly creating T using for loops seems nice)
    me_cur = initial_state_to_process['me_model']
    opp_cur = initial_state_to_process['opp_model']
    #TODO: input properly parameterized observation T.T, also fill out a and b -- otherwise this entire function is garbage
    a = [me_model((0.0,0.0),me_cur),0.0,0.0,0.0]
    b = []
    # first construct the transition matrix (so T_ij = P(i|j), where i could be cd for example)
    T = [[a[0] * b[0], a[1] * b[2], a[2] * b[1], a[3] * b[3]],
         [a[0] * (1 - b[0]), a[1] * (1 - b[2]), a[2] * (1 - b[1]), a[3] * (1 - b[3])],
         [(1 - a[0]) * b[0], (1 - a[1]) * b[2], (1 - a[2]) * b[1], (1 - a[3]) * b[3]],
         [(1 - a[0]) * (1 - b[0]), (1 - a[1]) * (1 - b[2]), (1 - a[2]) * (1 - b[1]), (1 - a[3]) * (1 - b[3])]]

    # next construct the reward vector associated with the transitions/new states
    r = tf.reshape(payoff_matrix,(4,1))

    #TODO: Note! This currently is asking the question "given that I'm currently in state s and have just received my reward, what's the value of being in s?"
    #Whereas normal value iteration asks "if I were to move to s, what would my expected future discounted reward be?"
    #In this second case, you get the eq v = r+gammaT^tv
    # finally solve the recursion for v (i.e. solve v = gammaT^tv+T^tr)
    A = tf.diag([1.0, 1.0, 1.0, 1.0]) - tf.multiply(tf.transpose(T), .9)
    v = tf.matmul(tf.linalg.inv(A), tf.matmul(tf.transpose(T), r))

    return v, T


# We can always add a random player, dynamics can be deterministic
def action_pair_dyn_model(_, me_action, them_action):
    return ActionPairStateNode(me_action,  them_action)

def simple_agent_model(observation, me_vars):
    prob_d = bound_probabilities(tf.multiply(me_vars.tf_node[0], observation.get_last_me_action_node().tf_node-0.5) +
                                 tf.multiply(me_vars.tf_node[1], observation.get_last_opp_action_node().tf_node-0.5) + me_vars.tf_node[2])
    return [1 - prob_d, prob_d]

def transparent_observation_model(state):
    return ActionPairObservationNode(state.get_last_x_action_node(), state.get_last_y_action_node())

def reverse_observation_model(state):
    return ActionPairObservationNode(state.get_last_y_action_node(), state.get_last_x_action_node())

def empty_update_model(me, _):
    return me


def make_agent(get_session, start_vector, payoff, name):
    last_me_action_node = InputNode(load_me_action)
    last_opp_action_node = InputNode(load_opp_action)
    opp_params_node = InputNode(load_opp_model)
    me_params_node = TriVal_VariableNode(start_vector, "me", get_session)

    inputs = [last_me_action_node, last_opp_action_node, opp_params_node]

    initial_state = ActionPairStateNode(last_me_action_node, last_opp_action_node)
    initial_state_to_process = TotalStateNode(initial_state, me_params_node,  opp_params_node,  0,  1.0)
    u = get_utility_node(get_utility_function_from_payoff(payoff), action_pair_dyn_model, simple_agent_model,
                         simple_agent_model, transparent_observation_model, reverse_observation_model,
                         empty_update_model, empty_update_model, initial_state_to_process, 1, False)

    return TransparentAgentDecorator(
                SamplingAgentDecorator(
                        NameAgentDecorator(
                            GradientDescentBasedAgent(
                                get_session,
                                simple_agent_model(ActionPairObservationNode(last_me_action_node, last_opp_action_node), me_params_node),
                                u,
                                me_params_node.tf_node,
                                load_inputs(inputs)
                            ),
                            name
                        )
                ),
                me_params_node.get_val
    )


# LOLA Sketch
# def opp_utility(me_action,opp_action):
#       #calculate utility node for opp

# def opp_update_model(opp, observation):
# opp_update = opp_vars+tf.train.GradientDescentOptimizer(0.01).minimize(1-opp_utility(me_model(observation,me_vars)
#                                       ,opp_model(observation,opp_vars))), var_list=[opp_vars]).get_gradients()
# get_session().run(opp_update, feed_dict=observation)
#  return opp

# TODO add an easy way to build "complete" policy spaces
# TODO add opponent update models that are more realistic (gradient descent based)
