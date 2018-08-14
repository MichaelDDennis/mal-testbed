from agents import *
from simulation import *
import tensorflow as tf
from typing import Any

# This produces a utility function, from probability distributions to
# note that we're not actually using this now
def get_mixed_utility_function(mat):
    def utility_function(state):

        a = state['me_action_node']
        b = state['opp_action_node']
        #TODO: check that this isn't backwards...
        #TODO: heck, check that changing this does anything o.O. It seems to not be doing anything...
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

        a = state['me_action_node']
        b = state['opp_action_node']

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

#TODO: test
def get_utility_node(reward_model, dyn_model, me_model, opp_model,
                     me_observation_model, opp_observation_model, me_update_model, opp_update_model,
                     initial_state_to_process, max_depth, markov_sim = False):

    states = []
    full_states_to_process = [initial_state_to_process]
    #TODO: prevent exponential blowup by taking advantage of Markov property (although this won't work with learning...)
    #TODO: also prevent exponential blowup in learning rate by renormalizing
    #TODO: Replace this with the exact calculation that involved the matrix inverse (this involves finding the transition matrix)
    for full_state in full_states_to_process:
        state = full_state['state']
        me_cur = full_state['me_model']
        opp_cur = full_state['opp_model']
        action_distr_me = me_model(me_observation_model(state), me_cur)
        action_distr_opp = opp_model(opp_observation_model(state), opp_cur)

        """
        if markov_sim:
            def compare_state(state_1,state_2):
                if ((state_1['me_action_node'] == state_2['me_action_node']) and (state_1['opp_action_node'] == state_2['opp_action_node'])):
                    return True
                else: return False
        """

        for action_me in range(2):
            for action_opp in range(2):
                new_state = dyn_model(state, action_me, action_opp)
                prob = tf.multiply(action_distr_me[action_me], action_distr_opp[action_opp])
                states.append((new_state, tf.multiply(full_state['prob'],prob)))
                #TODO: avoid exponential blowup using code like the following (but with a better equality check?)
                """
                if markov_sim:
                    for s in states:
                        if compare_state(s[0],new_state): #hopefully the equality check works >.<, naively using == didn't work
                            s[1] = s[1] + prob #ugh, illegal because s[1] is a tuple and can't be assigned to I guess
                        else:
                            states.append([new_state, prob])
                """
                if (full_state['depth'] < max_depth):
                    to_possibly_append = {'state': new_state,
                                                       'me_model': me_update_model(me_cur, me_observation_model(state)),
                                                       'opp_model': opp_update_model(opp_cur, opp_observation_model(state)),
                                                       'depth': full_state['depth']+1,
                                                        'prob': tf.multiply(full_state['prob'],prob)
                                                        }
                    if (not (markov_sim)):
                        full_states_to_process.append(to_possibly_append)
                    """
                    else:
                        for s in full_states_to_process:
                            if (compare_state(s['state'], new_state)) and (s['depth'] == full_state['depth']+1):
                                #increase s['prob']
                                pass
                            else:
                                full_states_to_process.append(to_possibly_append)
                    """

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
    return {'me_action_node': me_action, 'opp_action_node': them_action}

def simple_agent_model(observation, me_vars):
    prob_d = bound_probabilities(tf.multiply(me_vars[0], observation['me_action_node']-0.5) +
                                 tf.multiply(me_vars[1], observation['opp_action_node']-0.5) + me_vars[2])
    return [1 - prob_d, prob_d]

def transparent_observation_model(state):
    return state

def reverse_observation_model(state):
    return {'me_action_node': state['opp_action_node'], 'opp_action_node': state['me_action_node']}

def empty_update_model(me, _):
    return me

def make_agent(get_session, start_vector, payoff, name, type = "naive gradient", test = False):
    last_me = tf.placeholder(tf.float32)
    last_opp = tf.placeholder(tf.float32)
    opp = tf.placeholder(tf.float32, (3,))
    me = tf.Variable(start_vector, name="me")

    # TODO add an easy way to build "complete" policy spaces











    # def opp_utility(me_action,opp_action):
    #       #calculate utility node for opp

    # TODO add opponent update models that are more realistic (gradient descent based)
    #def opp_update_model(opp, observation):
        # opp_update = opp_vars+tf.train.GradientDescentOptimizer(0.01).minimize(1-opp_utility(me_model(observation,me_vars)
        #                                       ,opp_model(observation,opp_vars))), var_list=[opp_vars]).get_gradients()
        # get_session().run(opp_update, feed_dict=observation)
      #  return opp

    initial_state = {'me_action_node': last_me, 'opp_action_node': last_opp}
    initial_state_to_process = {'state': initial_state, 'me_model': me, 'opp_model': opp, 'depth': 0, 'prob': 1.0}
    u=get_utility_node(get_utility_function_from_payoff(payoff), action_pair_dyn_model, simple_agent_model, simple_agent_model,
                       transparent_observation_model, reverse_observation_model, empty_update_model, empty_update_model,
                       initial_state_to_process, 1, False)

    # TODO Make actions and observations into objects so you don't have to keep passing around hash maps
    # TODO add type checking
    def make_state(observation: ActionPairObservation[ModelActionPair[Any, ActionDistributionPair],
                                                      ModelActionPair[Any, ActionDistributionPair]]):
        return {opp: observation.get_last_opp_action().get_model(),
                last_me: observation.get_last_me_action().get_action().get_action(),
                last_opp: observation.get_last_opp_action().get_action().get_action()}

    def get_model():
        return get_session().run(me)

    if (type == "naive gradient"):
        return TransparentAgentDecorator(SamplingAgentDecorator(NameAgentDecorator(GradientDescentBasedAgent(
            get_session, simple_agent_model(transparent_observation_model(initial_state), me), u, me, make_state), name)), get_model)

    return TransparentAgentDecorator(SamplingAgentDecorator(NameAgentDecorator(ConstantStrategyAgent(
        get_session, simple_agent_model(transparent_observation_model(initial_state), me), u, me, make_state), name)), get_model)


def make_constant_agent(get_session, start_vector, name):
    last_me = tf.placeholder(tf.float32)
    last_opp = tf.placeholder(tf.float32)

    def me_model(me_vars):
        prob_d = bound_probabilities(tf.multiply(me_vars[0], last_me-0.5) +
                                     tf.multiply(me_vars[1], last_opp-0.5) + me_vars[2])
        return [1 - prob_d, prob_d]

    def make_state(observation: ActionPairObservation[ModelActionPair[Any, ActionDistributionPair],
                                                      ModelActionPair[Any, ActionDistributionPair]]):
        return {last_me: observation.get_last_me_action().get_action().get_action(),
                last_opp: observation.get_last_opp_action().get_action().get_action()}

    def get_model():
        return start_vector

    return TransparentAgentDecorator(SamplingAgentDecorator(NameAgentDecorator(ConstantStrategyAgent(
        get_session, me_model(start_vector)
        ,make_state), name)), get_model)


def initial_state_maker(me_action, opp_action, me_model, opp_model):
    return ActionPairState(ModelActionPair(me_model, ActionDistributionPair(me_action, [])),
                           ModelActionPair(opp_model, ActionDistributionPair(opp_action,[])))



# # By convention TensorFlowWrapperNodes will be immutable.  The underlying node should never change.
# class TensorFlowWrapperNode():
#
#
#     # Get the children tensor flow node wrappers for pretty print
#     def get_children(self):
#         pass
#
#
#     # Make sure they make string work
#     def ___str__(self):
#         pass
#
#
#     # Gives a function that will get the current values of this node given a session
#     def get_current_value_function(self):
#         pass
#
#
#     # Get the tensor flow node that this is wrapping
#     def get_tensor_flow_node(self):
#         pass
#
#
#     # Gets a list of tensor flow input nodes on which this nodes computation depends.
#     def get_placeholder_dependencies(self):
#         pass




# class ObservationNode():
#     pass
#
#
# class ActionNode():
#     pass
#
#
# class AgentNode():
#     pass
#
#
# class PredictionNode():
#     pass
#
#
# class MemoryNode():
#     pass
#
#
# class StateNode():
#     pass
#
#
# class UpdateNode():
#     def __init__(self, memoryNode : MemoryNode, update):
#
#     pass








