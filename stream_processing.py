import time


# This simply iterates through the states as they are produced from the simulation, prints them, and hands them to
# another stream
def print_state_decorator(simulation):
    for s in simulation:
        print(s)
        yield s


# This simply slows down the stream
def slow_sim_decorator(simulation, delay):
    for s in simulation:
        yield s
        time.sleep(delay)


# This runs the stream until the end
def run_sim(simulation):
    for _ in simulation:
        pass
