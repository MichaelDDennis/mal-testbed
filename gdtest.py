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


def main():
    global session
    initial_model_agent_a = [0.0, 0.0, -2.0]
    initial_model_agent_b = [0.0, 5.0, 0.0]
    initial_model_agent_titty = [0.0,1000000.0,0.0]
    initial_model_agent_cooperate_bot = [0.0,0.0,-10000.0]
    prisoners_payoff = [[400.0, 0.0],
                        [401.0, 50.0]]
    defection_is_magic = [[0.0, 100.0],
                        [300.0, 500.0]]
    #TODO: Figure out how to pass by value for Christ's sake -- any changes we make to agent_a keep getting perpetuated >.<
    agent_a = make_agent(get_session,initial_model_agent_a[:],prisoners_payoff,"Agent A", "naive gradient")
    agent_b = make_agent(get_session,initial_model_agent_b[:],prisoners_payoff, "Agent B", "naive gradient")
    agent_titty = make_agent(get_session,initial_model_agent_titty, prisoners_payoff, "Agent Tits", "naive gradient")
    agent_cooperate_bot = make_agent(get_session,initial_model_agent_cooperate_bot, prisoners_payoff, "Agent Cooperate Bot", "naive gradient")

    defection_confection = False
    if defection_confection:
        agent_a_d = make_agent(initial_model_agent_a[:], defection_is_magic, "Agent A")
        agent_b_d = make_agent(initial_model_agent_b[:], defection_is_magic, "Agent B")


    # Setting up tensor flow before running the simulation
    model = tf.global_variables_initializer()
    with tf.Session() as session:

        session.run(model)
        #uncomment write_test_sims() if you want to write a test suite
        #write_test_sims(initial_model_agent_a[:], initial_model_agent_b[:],agent_a,agent_b,"_prisoners")
        #uncomment compare_against_written_tests to test consistency
        #compare_against_written_tests(initial_model_agent_a[:], initial_model_agent_b[:],agent_a,agent_b,"_prisoners")

        testing_titties = True
        if testing_titties:
            #TODO: Get ConstantStrategyAgent to actually work >.<, so that you can call agent_titty = make_agent(initial_model_agent_tits, prisoners_payoff, "Agent Tits", "titty")
            initial_state = initial_state_maker(0.0, 0.0, initial_model_agent_a[:], initial_model_agent_titty)
            create_and_run_printy_sim(initial_state, action_pair_dynamics, full_observation_function,
                                     reflective_pair_observation_function, agent_a, agent_titty)
        testing_cooperate_bot = False
        if testing_cooperate_bot:
            initial_state = initial_state_maker(0.0, 0.0, initial_model_agent_a[:], initial_model_agent_cooperate_bot)
            create_and_run_printy_sim(initial_state, action_pair_dynamics, full_observation_function,
                                     reflective_pair_observation_function, agent_a, agent_cooperate_bot)

        #as a test, the agents should learn to defect (which is socially optimal with this payoff matrix lol)
        if defection_confection:
            initial_state = initial_state_maker(0.0, 0.0, initial_model_agent_a[:], initial_model_agent_b[:])
            create_and_run_printy_sim(initial_state, action_pair_dynamics, full_observation_function,
                                      reflective_pair_observation_function, agent_a_d, agent_b_d)

        initial_state = initial_state_maker(1.0,1.0,initial_model_agent_a[:],initial_model_agent_b[:])
        create_and_run_printy_sim(initial_state, action_pair_dynamics, full_observation_function,
                              reflective_pair_observation_function, agent_a, agent_b)

        initial_state = initial_state_maker(0.0,0.0,initial_model_agent_a[:],initial_model_agent_b[:])
        create_and_run_printy_sim(initial_state, action_pair_dynamics, full_observation_function,
                              reflective_pair_observation_function, agent_a, agent_b)



if __name__ == "__main__":
    main()
