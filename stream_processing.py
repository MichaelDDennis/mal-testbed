import time
from typing import TypeVar, Iterator, Generator, Any
from simulation import ActionPairState
from agents import ModelActionPair, ActionDistributionPair

T = TypeVar("T")


# This simply iterates through the states as they are produced from the simulation, prints them, and hands them to
# another stream
def print_state_decorator(simulation: Iterator[T]) -> Generator[T, None, None]:
    for s in simulation:
        print(s)
        yield s


# This simply slows down the stream
def slow_sim_decorator(simulation: Iterator[T], delay: int) -> Generator[T, None, None]:
    for s in simulation:
        yield s
        time.sleep(delay)


# This runs the stream until the end
def run_sim(simulation: Iterator[T]) -> None:
    for _ in simulation:
        pass


def print_actions(simulation: Iterator[ActionPairState[ModelActionPair[Any, ActionDistributionPair],
                                                       ModelActionPair[Any, ActionDistributionPair]]]) \
                  -> Generator[ActionPairState[ModelActionPair[Any, ActionDistributionPair],
                                               ModelActionPair[Any, ActionDistributionPair]], None, None]:
    for s in simulation:
        print("Actions were: ({},{})".format(s.get_last_x_action().get_action().get_action(),
                                             s.get_last_y_action().get_action().get_action()))
        yield s


def write_actions(simulation: Iterator[ActionPairState[ModelActionPair[Any, ActionDistributionPair],
                                                       ModelActionPair[Any, ActionDistributionPair]]], file) \
                  -> Generator[ActionPairState[ModelActionPair[Any, ActionDistributionPair],
                                               ModelActionPair[Any, ActionDistributionPair]], None, None]:
    for s in simulation:
        file.write("Actions were: ({},{})".format(s.get_last_x_action().get_action().get_action(),
                                                  s.get_last_y_action().get_action().get_action()))
        file.write("\n")
        yield s


def print_distributions(simulation: Iterator[ActionPairState[ModelActionPair[Any, ActionDistributionPair],
                                                             ModelActionPair[Any, ActionDistributionPair]]]) \
                        -> Generator[ActionPairState[ModelActionPair[Any, ActionDistributionPair],
                                                     ModelActionPair[Any, ActionDistributionPair]], None, None]:
    for s in simulation:
        print("Action Distributions were: ({},{})".format(s.get_last_x_action().get_action().get_distribution(),
                                                          s.get_last_y_action().get_action().get_distribution()))
        yield s


def write_distributions(simulation: Iterator[ActionPairState[ModelActionPair[Any, ActionDistributionPair],
                                                             ModelActionPair[Any, ActionDistributionPair]]], file) \
                        -> Generator[ActionPairState[ModelActionPair[Any, ActionDistributionPair],
                                                     ModelActionPair[Any, ActionDistributionPair]], None, None]:
    for s in simulation:
        file.write("Action Distributions were: ({},{})".format(s.get_last_x_action().get_action().get_distribution(),
                                                               s.get_last_y_action().get_action().get_distribution()))
        file.write("\n")
        yield s


def print_model(simulation: Iterator[ActionPairState[ModelActionPair, ModelActionPair]]) \
                -> Generator[ActionPairState[ModelActionPair, ModelActionPair], None, None]:
    for s in simulation:
        print("Model Parameters were: ({},{})".format(s.get_last_x_action().get_model(),
                                                      s.get_last_y_action().get_model()))
        yield s


def write_model(simulation: Iterator[ActionPairState[ModelActionPair, ModelActionPair]], file) \
                -> Generator[ActionPairState[ModelActionPair, ModelActionPair], None, None]:
    for s in simulation:
        file.write("Model Parameters were: ({},{})".format(s.get_last_x_action().get_model(),
                                                           s.get_last_y_action().get_model()))
        file.write("\n")
        yield s


# This should be on the inside
def print_count(simulation: Iterator[ActionPairState]) -> Generator[ActionPairState, None, None]:
    count = 0
    for s in simulation:
        print("Starting simulation round: {}".format(count))
        yield s
        count += 1
        print("")


def write_count(simulation: Iterator[ActionPairState], file) -> Generator[ActionPairState, None, None]:
    count = 0
    for s in simulation:
        file.write("Starting simulation round: {}".format(count))
        file.write("\n")
        yield s
        count += 1
        file.write("\n")
