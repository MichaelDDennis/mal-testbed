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


def print_actions(simulation):
    for s in simulation:
        print("Actions were: ({},{})".format(s['last_action_a']['action']['sample'],
                                             s['last_action_b']['action']['sample']))
        yield s

def write_actions(simulation,file):
    for s in simulation:
        file.write("Actions were: ({},{})".format(s['last_action_a']['action']['sample'],
                                             s['last_action_b']['action']['sample']))
        file.write("\n")
        yield s


def print_distributions(simulation):
    for s in simulation:
        print("Action Distributions were: ({},{})".format(s['last_action_a']['action']['distribution'],
                                                          s['last_action_b']['action']['distribution']))
        yield s

def write_distributions(simulation, file):
    for s in simulation:
        file.write("Action Distributions were: ({},{})".format(s['last_action_a']['action']['distribution'],
                                                          s['last_action_b']['action']['distribution']))
        file.write("\n")
        yield s


def print_model(simulation):
    for s in simulation:
        print("Model Parameters were: ({},{})".format(s['last_action_a']['model'],
                                                      s['last_action_b']['model']))
        yield s

def write_model(simulation, file):
    for s in simulation:
        file.write("Model Parameters were: ({},{})".format(s['last_action_a']['model'],
                                                      s['last_action_b']['model']))
        file.write("\n")
        yield s

# This should be on the inside
def print_count(simulation):
    count = 0
    for s in simulation:
        print("Starting simulation round: {}".format(count))
        yield s
        count += 1
        print("")

def write_count(simulation, file):
    count = 0
    for s in simulation:
        file.write("Starting simulation round: {}".format(count))
        file.write("\n")
        yield s
        count += 1
        file.write("\n")

