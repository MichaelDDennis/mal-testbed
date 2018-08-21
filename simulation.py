from agents import Agent
from typing import TypeVar, Generator, Callable, Generic


# State = NewType("State", object)
# X_Action = NewType("X_Action", object)
# Y_Action = NewType("Y_Action", object)
# X_Observation = NewType("X_Observation", object)
# Y_Observation = NewType("Y_Observation", object)

State = TypeVar("State")
X_Action = TypeVar("X_Action")
Y_Action = TypeVar("Y_Action")
X_Observation = TypeVar("X_Observation")
Y_Observation = TypeVar("Y_Observation")


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
def simulate(initial_state: State, dynamics: Callable[[State, X_Action, Y_Action], State],
             x_observation_function: Callable[[State], X_Observation],
             y_observation_function: Callable[[State], Y_Observation],
             x_agent: Agent[X_Observation, X_Action],
             y_agent: Agent[Y_Observation, Y_Action]) -> Generator[State, None, None]:

    state = initial_state
    for i in range(100000000):
        action_b = y_agent.get_action(y_observation_function(state))
        action_a = x_agent.get_action(x_observation_function(state))

        state = dynamics(state, action_a, action_b)

        yield state


class ActionPairState(Generic[X_Action, Y_Action]):

    def __init__(self, x_action_in: X_Action, y_action_in: Y_Action) -> None:
        self.x_action = x_action_in
        self.y_action = y_action_in

    def get_last_x_action(self) -> X_Action:
        return self.x_action

    def get_last_y_action(self) -> Y_Action:
        return self.y_action


class ActionPairObservation(Generic[X_Action, Y_Action]):

    def __init__(self, x_action_in: X_Action, y_action_in: Y_Action) -> None:
        self.x_action = x_action_in
        self.y_action = y_action_in

    def get_last_me_action(self) -> X_Action:
        return self.x_action

    def get_last_opp_action(self) -> Y_Action:
        return self.y_action


# State Dynamics which do not depend on the last state, and give a unique state for every action pair
def action_pair_dynamics(_, last_action_a: X_Action, last_action_b: Y_Action) -> ActionPairState[X_Action, Y_Action]:
    return ActionPairState(last_action_a, last_action_b)


# Observation Function for Fully Observable Environments
def full_observation_function(state: ActionPairState[X_Action, Y_Action]) -> ActionPairObservation[X_Action, Y_Action]:
    return ActionPairObservation(state.x_action, state.y_action)


# Reflects the observation so that the game is symmetric
def reflective_pair_observation_function(state: ActionPairState[X_Action, Y_Action]) \
                                         -> ActionPairObservation[Y_Action, X_Action]:
    return ActionPairObservation(state.y_action, state.x_action)
