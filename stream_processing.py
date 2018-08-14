import time
from typing import TypeVar, Iterator, Generator
from simulation import ActionPairState

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


def print_actions(simulation: Iterator[ActionPairState]) -> Generator[ActionPairState, None, None]:
    for s in simulation:
        print("Actions were: ({},{})".format(s.get_last_x_action()['action']['sample'],
                                             s.get_last_y_action()['action']['sample']))
        yield s


def write_actions(simulation: Iterator[ActionPairState], file) -> Generator[ActionPairState, None, None]:
    for s in simulation:
        file.write("Actions were: ({},{})".format(s.get_last_x_action()['action']['sample'],
                                                  s.get_last_y_action()['action']['sample']))
        file.write("\n")
        yield s


def print_distributions(simulation: Iterator[ActionPairState]) -> Generator[ActionPairState, None, None]:
    for s in simulation:
        print("Action Distributions were: ({},{})".format(s.get_last_x_action()['action']['distribution'],
                                                          s.get_last_y_action()['action']['distribution']))
        yield s


def write_distributions(simulation: Iterator[ActionPairState], file) -> Generator[ActionPairState, None, None]:
    for s in simulation:
        file.write("Action Distributions were: ({},{})".format(s.get_last_x_action()['action']['distribution'],
                                                               s.get_last_y_action()['action']['distribution']))
        file.write("\n")
        yield s


def print_model(simulation: Iterator[ActionPairState]) -> Generator[ActionPairState, None, None]:
    for s in simulation:
        print("Model Parameters were: ({},{})".format(s.get_last_x_action()['model'],
                                                      s.get_last_y_action()['model']))
        yield s


def write_model(simulation: Iterator[ActionPairState], file) -> Generator[ActionPairState, None, None]:
    for s in simulation:
        file.write("Model Parameters were: ({},{})".format(s.get_last_x_action()['model'],
                                                           s.get_last_y_action()['model']))
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
