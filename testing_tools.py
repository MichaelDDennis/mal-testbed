from gd_based_tools import *
from stream_processing import *

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
    simulation = print_count(simulation, f)

    simulation = print_actions(simulation, f)
    simulation = print_distributions(simulation, f)
    simulation = print_model(simulation, f)

    run_sim(simulation)
    f.close()

def write_test_sims(initial_model_agent_a,initial_model_agent_b,agent_a,agent_b, game):
    #writing each 1000-round sim takes about a minute

    #standard payoff matrix, initialized to cc
    initial_state = initial_state_maker(1.0, 1.0, initial_model_agent_a[:], initial_model_agent_b[:])
    create_and_write_sim("init_to_dd" + game, initial_state, action_pair_dynamics, full_observation_function,
                              reflective_pair_observation_function, agent_a, agent_b)
    #standard payoff matrix initialized to cc
    initial_state = initial_state_maker(0.0, 0.0, initial_model_agent_a[:], initial_model_agent_b[:])
    create_and_write_sim("init_to_cc" + game,initial_state, action_pair_dynamics, full_observation_function,
                              reflective_pair_observation_function, agent_a, agent_b)

def compare_against_written_tests(initial_model_agent_a, initial_model_agent_b,agent_a,agent_b,game):
    cc = open("init_to_cc"+game,'r')
    dd = open("init_to_dd"+game,'r')

    moves = {"dd":{"string":"dd","num":[1.0,1.0]},"cc":{"string":"cc","num":[0.0,0.0]}}
    for move in moves:
        initial_state = initial_state_maker(moves[move]["num"][0],moves[move]["num"][1], initial_model_agent_a[:], initial_model_agent_b[:])
        create_and_write_sim("init_to_"+moves[move]["string"]+"_update", initial_state, action_pair_dynamics, full_observation_function,
                                  reflective_pair_observation_function, agent_a, agent_b)
        old_file = open("init_to_"+moves[move]["string"]+game,'r')
        with open("init_to_"+moves[move]["string"]+"_update") as update:
            line_num = 0
            for update_line in update:
                old_line = old_file.readline()
                line_num = line_num + 1
                if (old_line != update_line):
                    raise Exception("Discrepancy between init_to_"+moves[move]["string"]+"_update and updated run on line __")
                    #^ugh, I'm an exception noob >.<
    cc.close()
    dd.close()

    print("Tests successful! You haven't broken anything with your most recent update ;)")