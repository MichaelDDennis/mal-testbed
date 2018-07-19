import tensorflow as tf
from simulation import simulate, action_pair_dynamics, full_observation_function, reflective_pair_observation_function
from agents import GradientDecentBasedAgent, TransparentAgentDecorator, SamplingAgentDecorator
from stream_processing import slow_sim_decorator, print_state_decorator, run_sim


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


def get_utility_of_states(reward, state_list):
    res = 0.0
    for state, prob in state_list:
        r = tf.multiply(reward(state), prob)
        res += r
    return res


def bound_probabilities(input_node):
    return tf.multiply(tf.tanh(input_node), 0.5)+0.5


def make_gradient_model(reward_model, dyn_model, me_model, opp_model,
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
                new_state = dyn_model(state, 1.0*action_me, 1.0*action_opp)
                prob = tf.multiply(action_distr_me[action_me], action_distr_opp[action_opp])
                states.append((new_state, prob))

                if full_state['depth'] < max_depth:
                    full_states_to_process.append({'state': new_state,
                                                   'me_model': me_update_model(me_cur, me_observation_model(state)),
                                                   'opp_model': opp_update_model(opp_cur, opp_observation_model(state)),
                                                   'depth': full_state['depth']+1})

    return get_utility_of_states(reward_model, states)


def make_agent(start_vector):
    last_me = tf.placeholder(tf.float32)
    last_opp = tf.placeholder(tf.float32)
    opp = tf.placeholder(tf.float32, (3,))
    me = tf.Variable(start_vector, name="me")

    payoff = [[400.0, 0.0],
              [401.0, 50.0]]

    # TODO add an easy way to build "complete" policy spaces

    def me_model(observation, me_vars):
        prob_c = bound_probabilities(tf.multiply(me_vars[0], observation['me_action_node']) +
                                     tf.multiply(me_vars[1], observation['opp_action_node']) + me_vars[2])
        return [prob_c, 1.0 - prob_c]

    def opp_model(observation, opp_vars):
        prob_c = bound_probabilities(tf.multiply(opp_vars[0], observation['opp_action_node']) +
                                     tf.multiply(opp_vars[1], observation['me_action_node']) + opp_vars[2])
        return [prob_c, 1.0 - prob_c]

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
    u=make_gradient_model(get_mixed_utility_function(payoff), dyn_model, me_model, opp_model,
                          me_observation_model, opp_observation_model, me_update_model, opp_update_model,
                          initial_state_to_process, 0)

    # TODO Make actions and observations into objects so you don't have to keep passing around hash maps
    # TODO add type checking
    def make_state(observation):
        return {opp: observation['last_action_b']['model'], last_me: observation['last_action_a']['action'],
                last_opp: observation['last_action_b']['action']}

    def get_model():
        return get_session().run(me)

    return TransparentAgentDecorator(SamplingAgentDecorator(GradientDecentBasedAgent(
        get_session, me_model(me_observation_model(initial_state), me), u, me, make_state)), get_model)


def main():
    global session
    agent_a = make_agent([100000.0, 0.0, 0.0])
    agent_b = make_agent([100000.0, 0.0, 0.0])

    # Setting up tensor flow before running the simulation
    model = tf.global_variables_initializer()
    with tf.Session() as session:

        session.run(model)

        initial_state = {'last_action_a': {'action': 1.0, 'model': [0.0, 100000.0, 0.0]},
                         'last_action_b': {'action': 1.0, 'model': [100000.0, 0.0, 0.0]}}
        simulation = simulate(initial_state, action_pair_dynamics, full_observation_function,
                              reflective_pair_observation_function, agent_a, agent_b)

        run_sim(print_state_decorator(slow_sim_decorator(simulation, 1)))


if __name__ == "__main__":
    main()
