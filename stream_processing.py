import time
from typing import TypeVar, Iterator, Generator, Any, TextIO
from simulation import ActionPairState
from agents import ModelActionPair, ActionDistributionPair
import sys

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


def skip_n(simulation: Iterator[T], n: int) -> Generator[T, None, None]:
    i = 0
    for s in simulation:
        if i//n == 0:
            yield s
        i = i+1

# This runs the stream until the end
def run_sim(simulation: Iterator[T]) -> None:
    for _ in simulation:
        pass


def print_actions(simulation: Iterator[ActionPairState[ModelActionPair[Any, ActionDistributionPair],
                                                       ModelActionPair[Any, ActionDistributionPair]]],
                  output: TextIO=sys.stdout ) \
                  -> Generator[ActionPairState[ModelActionPair[Any, ActionDistributionPair],
                                               ModelActionPair[Any, ActionDistributionPair]], None, None]:
    for s in simulation:
        output.write("Actions were: ({},{})\n".format(s.get_last_x_action().get_action().get_action(),
                                                      s.get_last_y_action().get_action().get_action()))
        output.flush()
        yield s


def print_distributions(simulation: Iterator[ActionPairState[ModelActionPair[Any, ActionDistributionPair],
                                                             ModelActionPair[Any, ActionDistributionPair]]],
                        output: TextIO=sys.stdout) \
                        -> Generator[ActionPairState[ModelActionPair[Any, ActionDistributionPair],
                                                     ModelActionPair[Any, ActionDistributionPair]], None, None]:
    for s in simulation:
        output.write("Action Distributions were: ({},{})\n".format(
                                                            s.get_last_x_action().get_action().get_distribution(),
                                                            s.get_last_y_action().get_action().get_distribution()))
        output.flush()
        yield s


def print_model(simulation: Iterator[ActionPairState[ModelActionPair, ModelActionPair]],
                output: TextIO=sys.stdout) \
                -> Generator[ActionPairState[ModelActionPair, ModelActionPair], None, None]:
    for s in simulation:
        output.write("Model Parameters were: ({},{})\n".format(s.get_last_x_action().get_model(),
                                                               s.get_last_y_action().get_model()))
        output.flush()
        yield s


# This should be on the inside
def print_count(simulation: Iterator[ActionPairState],
                output: TextIO=sys.stdout) -> Generator[ActionPairState, None, None]:
    count = 0
    for s in simulation:
        output.write("Starting simulation round: {}\n".format(count))
        output.flush()
        yield s
        count += 1
        output.write("\n")
        output.flush()
