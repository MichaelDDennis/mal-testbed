import tensorflow as tf
from testing_tools import *


# This is just to serve as a hook to allow for building agents before the session has started
def get_session():
    return session

def test_equals(x,y,s):
    if x != y:
        raise Exception("{} was not the same, should be {} but was {}".format(s,x,y))

def get_utility_node_test():
    #Easy Tests
    test_equals(50.0,get_utility_node_test_helper([0.0, 0.0, 10000.0], [0.0, 0.0, 10000.0]),"DD Test on third param")
    test_equals(0.0, get_utility_node_test_helper([0.0, 0.0, -10000.0], [0.0, 0.0, 10000.0]), "CD Test on third param")
    test_equals(500.0, get_utility_node_test_helper([0.0, 0.0, 10000.0], [0.0, 0.0, -10000.0]), "DC Test on third param")
    test_equals(400.0, get_utility_node_test_helper([0.0, 0.0, -10000.0], [0.0, 0.0, -10000.0]), "CC Test on third param")
    test_equals(50.0,get_utility_node_test_helper([-10000.0, 0.0, 0.0], [-10000.0, 0.0, 0.0]),"DD Test on first param")
    test_equals(0.0, get_utility_node_test_helper([100000.0, 0.0, 0.0], [-10000.0, 0.0, 0.0]), "CD Test on first param")
    test_equals(500.0, get_utility_node_test_helper([-10000.0, 0.0, 0.0], [100000.0, 0.0, 0.0]), "DC Test on first param")
    test_equals(400.0, get_utility_node_test_helper([100000.0, 0.0, 0.0], [100000.0, 0.0, 0.0]), "CC Test on first param")
    test_equals(50.0,get_utility_node_test_helper([ 0.0, -10000.0, 0.0], [0.0, -10000.0, 0.0]),"DD Test on second param")
    test_equals(0.0, get_utility_node_test_helper([0.0, 100000.0, 0.0], [0.0, -10000.0, 0.0]), "CD Test on second param")
    test_equals(500.0, get_utility_node_test_helper([0.0, -10000.0, 0.0], [0.0, 100000.0, 0.0]), "DC Test on second param")
    test_equals(400.0, get_utility_node_test_helper([0.0, 100000.0, 0.0], [0.0, 100000.0, 0.0]), "CC Test on second param")

    #Harder Tests
    test_equals(25.0, get_utility_node_test_helper([20000.0, 0.0, 10000.0], [0.0, 0.0, 10000.0]), "50/50 for p1, d for p2")
    test_equals(100.0, get_utility_node_test_helper([0.0, 0.0, 10000.0], [0.0, 0.0, 10000.0], 1), "DD depth 1")
    test_equals(62.5, get_utility_node_test_helper([20000.0, 0.0, 10000.0], [0.0, 0.0, 10000.0], 1), "50/50 for p1, d for p2, depth 1")

    #Regression Tests
    test_equals(200.0,get_utility_node_test_helper([0.0, 0.0, 10000.0], [0.0, 0.0, 10000.0],3),"DD Test on third param")
    test_equals(0.0, get_utility_node_test_helper([0.0, 0.0, -10000.0], [0.0, 0.0, 10000.0],2), "CD Test on third param")
    test_equals(1000.0, get_utility_node_test_helper([0.0, 0.0, 10000.0], [0.0, 0.0, -10000.0],1), "DC Test on third param")
    test_equals(1200.0, get_utility_node_test_helper([0.0, 0.0, -10000.0], [0.0, 0.0, -1000.0],2), "CC Test on third param")
    test_equals(500.0,get_utility_node_test_helper([-10000.0, 0.0, 0.0], [-10000.0, 0.0, 0.0],2),"DD Test on first param")
    test_equals(400.0, get_utility_node_test_helper([100000.0, 0.0, 0.0], [-10000.0, 0.0, 0.0],2), "CD Test on first param")
    test_equals(1400.0, get_utility_node_test_helper([-10000.0, 0.0, 0.0], [100000.0, 0.0, 0.0],2), "DC Test on first param")
    test_equals(1200.0, get_utility_node_test_helper([100000.0, 0.0, 0.0], [100000.0, 0.0, 0.0],2), "CC Test on first param")
    test_equals(500.0,get_utility_node_test_helper([ 0.0, -10000.0, 0.0], [0.0, -10000.0, 0.0],2),"DD Test on second param")
    test_equals(550.0, get_utility_node_test_helper([0.0, 100000.0, 0.0], [0.0, -10000.0, 0.0],2), "CD Test on second param")
    test_equals(550.0, get_utility_node_test_helper([0.0, -10000.0, 0.0], [0.0, 100000.0, 0.0],2), "DC Test on second param")
    test_equals(1200.0, get_utility_node_test_helper([0.0, 100000.0, 0.0], [0.0, 100000.0, 0.0],2), "CC Test on second param")


def get_utility_node_test_helper(a_params,b_params, depth=0):
    global session
    prisoners_payoff = [[400.0, 0.0],
                        [500.0, 50.0]]
    util_fun_node = get_utility_function_from_payoff(prisoners_payoff)


    # initial_model_agent_a = [.25, 0.5, -.3333333]
    # initial_model_agent_tit_for_tat = [.125, 0.25, -.166666666]


    last_me = tf.placeholder(tf.float32)
    last_opp = tf.placeholder(tf.float32)
    opp = tf.placeholder(tf.float32, (3,))
    me = tf.Variable(a_params, name="me")
    initial_state = {'me_action_node': last_me, 'opp_action_node': last_opp}
    initial_state_to_process = {'state': initial_state, 'me_model': me, 'opp_model': opp, 'depth': 0, 'prob': 1.0}
    u = get_utility_node(util_fun_node, action_pair_dyn_model, simple_agent_model,
                         simple_agent_model,
                         transparent_observation_model, reverse_observation_model, empty_update_model,
                         empty_update_model,
                         initial_state_to_process, depth, False)
#    u_a = get_utility_node(util_fun_node,action_pair_dyn_model,)

    # u_a_tensor_elts = []
    # for i in range(len(u_a)):
    #     u_a_tensor_elts.append(tf.Variable(u_a[i]))
    #agent_tit_for_tat = make_agent(get_session, initial_model_agent_tit_for_tat, prisoners_payoff, "Agent tit_for_tat")

    model = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(model)

        # initial_state = initial_state_maker(0.0, 0.0, initial_model_agent_a[:], initial_model_agent_tit_for_tat)
        # p = session.run(prisoners_payoff)
        # print(p)
        # u_a_0_val = session.run(u_a[0])
        # print(u_a_0_val)
        # u_a_val = session.run(u_a)
        # print(u_a_val)
        u_a_val = session.run(u, feed_dict={last_me: 0.0, last_opp: 0.0, opp: b_params }).item()
        return u_a_val

        """
        #recreate simulation code for fine-grained testing
        state = initial_state
        for i in range(1000):
            action_tit_for_tat = agent_tit_for_tat.get_action(full_observation_function(state))
            action_a = agent_a.get_action(full_observation_function(state))

            state = action_pair_dynamics(state, action_a, action_tit_for_tat)

            yield state
        """
    """
    u code
    reward_model, dyn_model, me_model, opp_model,
    me_observation_model, opp_observation_model, me_update_model, opp_update_model,
    initial_state_to_process, max_depth, markov_sim = False
    """

def bound_probabilities_test():
    global session
    prob = tf.placeholder(tf.float32)
    bound_prob = bound_probabilities(prob)

    model = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(model)
        bound_prob_val = session.run(bound_prob, feed_dict={prob: 3.0}).item()

        #print("The following value should be between 0 and 1: {}".format(bound_prob_val))

        if bound_prob_val < 0:
            raise Exception("Bounded probability was negative: {}".format(bound_prob_val))

        if bound_prob_val > 1:
            raise Exception("Bounded probability was above 1: {}".format(bound_prob_val))

        if bound_prob_val != 0.9975273609161377:
            raise Exception("Bounded probability is not what it used to be: {}".format(bound_prob_val))


def defection_confection_test():
    global session

    defection_is_magic = [[0.0, 100.0],
                          [300.0, 500.0]]
    initial_model_agent_a = [0.0, 0.0, -2.0]
    initial_model_agent_b = [0.0, 5.0, 0.0]
    agent_a = make_agent(get_session, initial_model_agent_a[:], defection_is_magic, "Agent A")
    agent_b = make_agent(get_session, initial_model_agent_b[:], defection_is_magic, "Agent B")

    model = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(model)

        initial_state = initial_state_maker(0.0, 0.0, initial_model_agent_a[:], initial_model_agent_b)
        create_and_run_printy_sim(initial_state, action_pair_dynamics, full_observation_function,
                                  reflective_pair_observation_function, agent_a, agent_b)

def cooperate_bot_test():
    prisoners_payoff = [[400.0, 0.0],
                        [401.0, 50.0]]
    initial_model_agent_a = [0.0, 0.0, -2.0]
    initial_model_agent_cooperate_bot = [0.0, 0.0, -10000.0]
    agent_a = make_agent(get_session, initial_model_agent_a[:], prisoners_payoff, "Agent A")
    agent_cooperate_bot = make_constant_agent(get_session, initial_model_agent_cooperate_bot, "Agent Cooperate Bot")

    global session
    model = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(model)

        initial_state = initial_state_maker(0.0, 0.0, initial_model_agent_a[:], initial_model_agent_cooperate_bot)
        create_and_run_printy_sim(initial_state, action_pair_dynamics, full_observation_function,
                                  reflective_pair_observation_function, agent_a, agent_cooperate_bot)

def tit_for_tat_bot_test():
    prisoners_payoff = [[400.0, 0.0],
                        [401.0, 50.0]]
    initial_model_agent_a = [0.0, 0.0, -2.0]
    initial_model_agent_tit_for_tat = [0.0, 1000000.0, 0.0]
    agent_a = make_agent(get_session, initial_model_agent_a[:], prisoners_payoff, "Agent A")
    agent_tit_for_tat = make_agent(get_session, initial_model_agent_tit_for_tat, prisoners_payoff, "Agent tit_for_tat")
    global session
    model = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(model)

        # TODO: Get ConstantStrategyAgent to actually work >.<, so that you can call agent_tit_for_tat = make_agent(get_session, initial_model_agent_tits, prisoners_payoff, "Agent Tits", "tit_for_tat")
        initial_state = initial_state_maker(0.0, 0.0, initial_model_agent_a[:], initial_model_agent_tit_for_tat)
        create_and_run_printy_sim(initial_state, action_pair_dynamics, full_observation_function,
                                  reflective_pair_observation_function, agent_a, agent_tit_for_tat)

def init_to_TFT_test():
    prisoners_payoff = [[400.0, 0.0],
                        [401.0, 50.0]]
    initial_model_agent_a = [0.0, 1000000.0, 0.0]
    initial_model_agent_b = [0.0, 1000000.0, 0.0]
    agent_a = make_agent(get_session, initial_model_agent_a[:], prisoners_payoff, "Agent A")
    agent_b = make_agent(get_session, initial_model_agent_b, prisoners_payoff, "Agent B")
    # TODO: Get ConstantStrategyAgent to actually work >.<, so that you can call agent_tit_for_tat = make_agent(get_session, initial_model_agent_tits, prisoners_payoff, "Agent Tits", "tit_for_tat")
    global session
    model = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(model)

        initial_state = initial_state_maker(0.0, 0.0, initial_model_agent_a[:], initial_model_agent_b)
        create_and_run_printy_sim(initial_state, action_pair_dynamics, full_observation_function,
                                  reflective_pair_observation_function, agent_a, agent_b)

#TODO: add double_cooperate_bot_test(), defect_bot_test(), init_to_zero_test() etc.
#TODO: (old issue, but still good to know) Figure out how to pass by value -- any changes we make to agent_a keep getting perpetuated >.<
#TODO: add written tests like:
    # #write_test_sims(initial_model_agent_a[:], initial_model_agent_b[:],agent_a,agent_b,"_prisoners")
    #compare_against_written_tests(initial_model_agent_a[:], initial_model_agent_b[:],agent_a,agent_b,"_prisoners")


def main():

    # bound_probabilities_test()

    # get_utility_node_test()

    tit_for_tat_bot_test()

    # cooperate_bot_test()

    # defection_confection_test()


if __name__ == "__main__":
    main()
