import tensorflow as tf
from simulation import simulate, action_pair_dynamics, full_observation_function, reflective_pair_observation_function
from agents import *
from stream_processing import *
from gd_based_tools import *
from copy import deepcopy
from testing_tools import *

# This is just to serve as a hook to allow for building agents before the session has started
def get_session():
    return session

def defection_confection_test():
    defection_is_magic = [[0.0, 100.0],
                          [300.0, 500.0]]
    initial_model_agent_a = [0.0, 0.0, -2.0]
    initial_model_agent_b = [0.0, 5.0, 0.0]
    agent_a = make_agent(get_session, initial_model_agent_a[:], defection_is_magic, "Agent A")
    agent_b = make_agent(get_session, initial_model_agent_b[:], defection_is_magic, "Agent B")
    global session
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
    agent_cooperate_bot = make_agent(get_session, initial_model_agent_cooperate_bot, prisoners_payoff,
                                     "Agent Cooperate Bot", "naive gradient")
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
    initial_model_agent_titty = [0.0, 1000000.0, 0.0]
    agent_a = make_agent(get_session, initial_model_agent_a[:], prisoners_payoff, "Agent A")
    agent_titty = make_agent(get_session, initial_model_agent_titty, prisoners_payoff, "Agent Titty")
    global session
    model = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(model)

        # TODO: Get ConstantStrategyAgent to actually work >.<, so that you can call agent_titty = make_agent(get_session, initial_model_agent_tits, prisoners_payoff, "Agent Tits", "titty")
        initial_state = initial_state_maker(0.0, 0.0, initial_model_agent_a[:], initial_model_agent_titty)
        create_and_run_printy_sim(initial_state, action_pair_dynamics, full_observation_function,
                                  reflective_pair_observation_function, agent_a, agent_titty)

def init_to_TFT_test():
    prisoners_payoff = [[400.0, 0.0],
                        [401.0, 50.0]]
    initial_model_agent_a = [0.0, 1000000.0, 0.0]
    initial_model_agent_b = [0.0, 1000000.0, 0.0]
    agent_a = make_agent(get_session, initial_model_agent_a[:], prisoners_payoff, "Agent A")
    agent_b = make_agent(get_session, initial_model_agent_b, prisoners_payoff, "Agent B")
    # TODO: Get ConstantStrategyAgent to actually work >.<, so that you can call agent_titty = make_agent(get_session, initial_model_agent_tits, prisoners_payoff, "Agent Tits", "titty")
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

    tit_for_tat_bot_test()

    # cooperate_bot_test()

    # defection_confection_test()


if __name__ == "__main__":
    main()
