import tensorflow as tf
from simulation import simulate, action_pair_dynamics, full_observation_function, reflective_pair_observation_function
from agents import *
from stream_processing import *
from test_suite import *
from copy import deepcopy

# This is just to serve as a hook to allow for building agents before the session has started
def get_session():
    return session


# This produces a utility function, from probability distributions to
def get_mixed_utility_function(mat):
    def utility_function(state):

        a = state['me_action_node']
        b = state['opp_action_node']
        pa = [a, 1.0 - a]
        pb = [b, 1.0 - b]

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


def get_utility_node(reward_model, dyn_model, me_model, opp_model,
                     me_observation_model, opp_observation_model, me_update_model, opp_update_model,
                     initial_state_to_process, max_depth):

    states = []
    full_states_to_process = [initial_state_to_process]
    for full_state in full_states_to_process:
        state = full_state['state']
        me_cur = full_state['me_model']
        opp_cur = full_state['opp_model']
        action_distr_me = me_model(me_observation_model(state), me_cur)
        action_distr_opp = opp_model(opp_observation_model(state), opp_cur)

        for action_me in range(2):
            for action_opp in range(2):
                new_state = dyn_model(state, action_me, action_opp)
                prob = tf.multiply(action_distr_me[action_me], action_distr_opp[action_opp])
                states.append((new_state, prob))

                if full_state['depth'] < max_depth:
                    full_states_to_process.append({'state': new_state,
                                                   'me_model': me_update_model(me_cur, me_observation_model(state)),
                                                   'opp_model': opp_update_model(opp_cur, opp_observation_model(state)),
                                                   'depth': full_state['depth']+1})

    return get_utility_of_states(reward_model, states)


def make_agent(start_vector, payoff, name):
    last_me = tf.placeholder(tf.float32)
    last_opp = tf.placeholder(tf.float32)
    opp = tf.placeholder(tf.float32, (3,))
    me = tf.Variable(start_vector, name="me")



    # TODO add an easy way to build "complete" policy spaces

    def me_model(observation, me_vars):
        prob_d = bound_probabilities(tf.multiply(me_vars[0], observation['me_action_node']-0.5) +
                                     tf.multiply(me_vars[1], observation['opp_action_node']-0.5) + me_vars[2])
        return [1 - prob_d, prob_d]

    def opp_model(observation, opp_vars):
        prob_d = bound_probabilities(tf.multiply(opp_vars[0], observation['me_action_node']-0.5) +
                                     tf.multiply(opp_vars[1], observation['opp_action_node']-0.5) + opp_vars[2])
        return [1 - prob_d, prob_d]

    # We can always add a random player, dynamics can be deterministic
    def dyn_model(_, me_action, them_action):
        return {'me_action_node': me_action, 'opp_action_node': them_action}

    def me_observation_model(state):
        return state

    def opp_observation_model(state):
        return {'me_action_node': state['opp_action_node'], 'opp_action_node': state['me_action_node']}

    def me_update_model(me, _):
        return me

    # TODO add opponent update models that are more realistic (gradient decent based)
    def opp_update_model(opp, _):
        return opp

    initial_state = {'me_action_node': last_me, 'opp_action_node': last_opp}
    initial_state_to_process = {'state': initial_state, 'me_model': me, 'opp_model': opp, 'depth': 0}
    u=get_utility_node(get_utility_function_from_payoff(payoff), dyn_model, me_model, opp_model,
                       me_observation_model, opp_observation_model, me_update_model, opp_update_model,
                       initial_state_to_process, 1)

    # TODO Make actions and observations into objects so you don't have to keep passing around hash maps
    # TODO add type checking
    def make_state(observation):
        return {opp: observation['last_action_b']['model'], last_me: observation['last_action_a']['action']['sample'],
                last_opp: observation['last_action_b']['action']['sample']}

    def get_model():
        return get_session().run(me)

    return TransparentAgentDecorator(SamplingAgentDecorator(NameAgentDecorator(GradientDescentBasedAgent(
        get_session, me_model(me_observation_model(initial_state), me), u, me, make_state), name)), get_model)

def initial_state_maker(me_action,opp_action,me_model,opp_model):
    return {'last_action_a': {'action': {'sample': me_action, 'distribution': []},
                                           'model': me_model},
                         'last_action_b': {'action': {'sample': opp_action, 'distribution': []},
                                           'model': opp_model}}

def create_and_run_printy_sim(initial_state, action_pair_dynamics, full_observation_function,
                              reflective_pair_observation_function, agent_a, agent_b):

    simulation = simulate(initial_state, action_pair_dynamics, full_observation_function,
                              reflective_pair_observation_function, agent_a, agent_b)
    simulation = print_count(simulation)
    simulation = slow_sim_decorator(simulation, 1)
    simulation = print_actions(simulation)
    simulation = print_distributions(simulation)
    simulation = print_model(simulation)

    run_sim(simulation)

def create_and_write_sim(file_name, initial_state, action_pair_dynamics, full_observation_function,
                              reflective_pair_observation_function, agent_a, agent_b):
    f = open(file_name, 'w')
    simulation = simulate(initial_state, action_pair_dynamics, full_observation_function,
                              reflective_pair_observation_function, agent_a, agent_b)
    simulation = write_count(simulation, f)

    simulation = write_actions(simulation, f)
    simulation = write_distributions(simulation, f)
    simulation = write_model(simulation, f)

    run_sim(simulation)
    f.close()

def write_test_sims(initial_model_agent_a, initial_model_agent_b,agent_a,agent_b):
    #writing each 1000-round sim takes about a minute

    #standard payoff matrix, initialized to cc
    initial_state = initial_state_maker(1.0, 1.0, initial_model_agent_a[:], initial_model_agent_b[:])
    create_and_write_sim("init_to_cc", initial_state, action_pair_dynamics, full_observation_function,
                              reflective_pair_observation_function, agent_a, agent_b)
    #standard payoff matrix initialized to cc
    initial_state = initial_state_maker(0.0, 0.0, initial_model_agent_a[:], initial_model_agent_b[:])
    create_and_write_sim("init_to_dd",initial_state, action_pair_dynamics, full_observation_function,
                              reflective_pair_observation_function, agent_a, agent_b)
    #payoff matrix with dd socially optimal, initialized to cc

def compare_against_written_tests(initial_model_agent_a, initial_model_agent_b,agent_a,agent_b):
    #TODO: refactor so you don't have duplicated code
    cc = open("init_to_cc",'r')
    dd = open("init_to_dd",'r')

    initial_state = initial_state_maker(1.0, 1.0, initial_model_agent_a[:], initial_model_agent_b[:])
    create_and_write_sim("init_to_cc_update", initial_state, action_pair_dynamics, full_observation_function,
                              reflective_pair_observation_function, agent_a, agent_b)
    with open("init_to_cc_update") as cc_update:
        line_num = 0
        for cc_update_line in cc_update:
            cc_line = cc.readline()
            line_num = line_num + 1
            if (cc_line != cc_update_line):
                raise Exception('Discrepancy between init_to_cc and updated run on line __')
                #^ugh, I'm an exception noob >.<
    cc.close()

    initial_state = initial_state_maker(0.0, 0.0, initial_model_agent_a[:], initial_model_agent_b[:])
    create_and_write_sim("init_to_dd_update",initial_state, action_pair_dynamics, full_observation_function,
                              reflective_pair_observation_function, agent_a, agent_b)
    dd_update = open("init_to_dd_update",'r')
    with open("init_to_dd_update") as dd_update:
        line_num = 0
        for dd_update_line in dd_update:
            dd_line = dd.readline()
            line_num = line_num + 1
            if (dd_line != dd_update_line):
                raise Exception('Discrepancy between init_to_dd and updated run on line __')
    dd.close()

    print("Tests successful! You haven't broken anything with your most recent update ;)")

def main():
    global session
    initial_model_agent_a = [0.0, 5.0, 0.0]
    initial_model_agent_b = [0.0, 5.0, 0.0]
    prisoners_payoff = [[400.0, 0.0],
                        [401.0, 50.0]]
    #TODO: Figure out how to pass by value for Christ's sake -- any changes we make to agent_a keep getting perpetuated >.<
    agent_a = make_agent(initial_model_agent_a[:],prisoners_payoff,"Agent A")
    agent_b = make_agent(initial_model_agent_b[:],prisoners_payoff, "Agent B")


    # Setting up tensor flow before running the simulation
    model = tf.global_variables_initializer()
    with tf.Session() as session:

        session.run(model)
        #uncomment write_test_sims() if you want to write a test suite
        #write_test_sims(initial_model_agent_a[:], initial_model_agent_b[:],agent_a,agent_b)
        #uncomment compare_against_written_tests to test consistency
        #compare_against_written_tests(initial_model_agent_a[:], initial_model_agent_b[:],agent_a,agent_b)

        initial_state = initial_state_maker(1.0,1.0,initial_model_agent_a[:],initial_model_agent_b[:])
        create_and_run_printy_sim(initial_state, action_pair_dynamics, full_observation_function,
                              reflective_pair_observation_function, agent_a, agent_b)

        initial_state = initial_state_maker(0.0,0.0,initial_model_agent_a[:],initial_model_agent_b[:])
        create_and_run_printy_sim(initial_state, action_pair_dynamics, full_observation_function,
                              reflective_pair_observation_function, agent_a, agent_b)



if __name__ == "__main__":
    main()
