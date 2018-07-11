from agents import Agent


# Simulate is the Main function we will be studying, the structure here is meant to force isolation between
# different parts of the code base so it is easier to tell what is going on and who knows about who.
#
# One specific way in which this helps avoid confusion is in the separating the game itself from how they
# update their models.  Though both will have similar structure, one will be in the agents head and the
# other will be in the real world.  Only the first needs to be differentiable and they need not always match.
#
# I also made this a streaming iterator so that interaction with it is easier and analysis of results can be seen
# as soon as possible.
#
# For clarity, the types of all of the arguments are as follows:
#
# Initial State: State
# Dyn: State x ActA x ActB --> State
# ObsA: State -> ObsA
# ObsB: State -> ObsB
#
# A: Agent
# B: Agent
#
def simulate(initial_state, dynamics, observation_function_a, observation_function_b,
             agent_a: Agent, agent_b: Agent):
    # Normal TensorFlow - initialize values, create a session and run the model
    state = initial_state

    for i in range(1000):
        action_b = agent_b.get_action(observation_function_b(state))
        action_a = agent_a.get_action(observation_function_a(state))

        state = dynamics(state, action_a, action_b)

        yield state


# State Dynamics which do not depend on the last state, and give a unique state for every action pair
def action_pair_dynamics(_, last_action_a, last_action_b):
    return {'last_action_a': last_action_a, 'last_action_b': last_action_b}


# Observation Function for Fully Observable Environments
def full_observation_function(state):
    return state


# Reflects the observation so that the game is symmetric
def reflective_pair_observation_function(state):
    res = {'last_action_a': state['last_action_b'], 'last_action_b': state['last_action_a']}
    return res
