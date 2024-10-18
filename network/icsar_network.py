from typing import Self
from networkx import Graph
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from enum import Enum

class State(Enum):
    IGNORANT = 1
    IGNORANT_REMOVAL = 2
    RUMOUR_ADVOCATE = 3
    RUMOUR_SPREADER = 4
    RUMOUR_CARRIER = 5
    TRUTH_CARRIER = 6
    TRUTH_SPREADER = 7
    TRUTH_ADVOCATE = 8

class ICSARModel:
    def __init__(self: Self, graph: Graph) -> None:
        self.graph = graph
        
        # Initialize states
        self.states = {node: State.IGNORANT for node in graph.nodes}
        initial_rumour_advocates = self.pick_random_ignorant_nodes(INITIAL_RUMOUR_ADVOCATES)
        for node in initial_rumour_advocates:
            self.states[node] = State.RUMOUR_ADVOCATE
        
        # History tracking
        self.i_history = []
        self.ir_history = []
        self.ra_history = []
        self.rs_history = []
        self.rc_history = []
        self.tc_history = []
        self.ts_history = []
        self.ta_history = []

    def pick_random_ignorant_nodes(self: Self, number: int) -> int:
        ignorant_nodes = [node for node, state in self.states.items() if state == State.IGNORANT or state == State.IGNORANT_REMOVAL]
        return np.random.choice(ignorant_nodes, number, replace=False)

    def step(self: Self) -> None:
        new_states = self.states.copy()
        
        for node in self.graph.nodes:
            state = self.states[node]
            if state == State.RUMOUR_SPREADER or state == State.RUMOUR_ADVOCATE:
                neighbors = list(self.graph.neighbors(node))

                if state == State.RUMOUR_SPREADER and np.random.random() < RUMOUR_SPREADER_TO_ADVOCATE_PROB:
                    new_states[node] = State.RUMOUR_ADVOCATE

                for neighbor in neighbors:
                    # If neighbour is ignorant
                    if self.states[neighbor] == State.IGNORANT:
                        if np.random.random() < RUMOUR_ACCEPTANCE_PROB:
                            new_states[neighbor] = State.RUMOUR_CARRIER
                        elif np.random.random() < IGNORE_PROB:
                            new_states[neighbor] = State.IGNORANT_REMOVAL
                    # If neighbour is rumour carrier
                    elif self.states[neighbor] == State.RUMOUR_CARRIER:
                        if np.random.random() < RUMOUR_CARRIER_TO_SPREADER_PROB:
                            new_states[neighbor] = State.RUMOUR_SPREADER
                    # If neighbour is truth carrier
                    elif self.states[neighbor] == State.TRUTH_CARRIER:
                        if np.random.random() < TRUTH_CARRIER_TO_RUMOUR_PROB:
                            new_states[neighbor] = State.RUMOUR_CARRIER
                    # If neighbour is truth spreader
                    elif self.states[neighbor] == State.TRUTH_SPREADER:
                        if np.random.random() < TRUTH_SPREADER_TO_RUMOUR_PROB:
                            new_states[neighbor] = State.RUMOUR_CARRIER
            elif state == State.TRUTH_SPREADER or state == State.TRUTH_ADVOCATE:
                neighbors = list(self.graph.neighbors(node))

                if state == State.TRUTH_SPREADER and np.random.random() < TRUTH_SPREADER_TO_ADVOCATE_PROB:
                    new_states[node] = State.TRUTH_ADVOCATE

                for neighbor in neighbors:
                    # If neighbour is ignorant
                    if self.states[neighbor] == State.IGNORANT:
                        if np.random.random() < TRUTH_ACCEPTANCE_PROB:
                            new_states[neighbor] = State.TRUTH_CARRIER
                        elif np.random.random() < IGNORE_PROB:
                            new_states[neighbor] = State.IGNORANT_REMOVAL
                    # If neighbour is truth carrier
                    elif self.states[neighbor] == State.TRUTH_CARRIER:
                        if np.random.random() < TRUTH_CARRIER_TO_SPREADER_PROB:
                            new_states[neighbor] = State.TRUTH_SPREADER
                    # If neighbour is rumour carrier
                    elif self.states[neighbor] == State.RUMOUR_CARRIER:
                        if np.random.random() < RUMOUR_CARRIER_TO_TRUTH_PROB:
                            new_states[neighbor] = State.TRUTH_CARRIER
                    # If neighbour is rumour spreader
                    elif self.states[neighbor] == State.RUMOUR_SPREADER:
                        if np.random.random() < RUMOUR_SPREADER_TO_TRUTH_PROB:
                            new_states[neighbor] = State.TRUTH_CARRIER
        
        self.states = new_states
        
        # Count states
        self.i_history.append(list(self.states.values()).count(State.IGNORANT))
        self.ir_history.append(list(self.states.values()).count(State.IGNORANT_REMOVAL))
        self.ra_history.append(list(self.states.values()).count(State.RUMOUR_ADVOCATE))
        self.rs_history.append(list(self.states.values()).count(State.RUMOUR_SPREADER))
        self.rc_history.append(list(self.states.values()).count(State.RUMOUR_CARRIER))
        self.tc_history.append(list(self.states.values()).count(State.TRUTH_CARRIER))
        self.ts_history.append(list(self.states.values()).count(State.TRUTH_SPREADER))
        self.ta_history.append(list(self.states.values()).count(State.TRUTH_ADVOCATE))

    def draw(self: Self) -> None:
        plt.figure(figsize=(10, 6))

        # Assemble colour map
        colours = []
        for node in self.graph.nodes:
            state = self.states[node]
            colours.append(STATUS_COLOURS[state])

        nx.draw(self.graph, node_size=50, node_color=colours, with_labels=False)
        plt.title('ICSAR Model of Rumor Propagation on a Graph')
        plt.show()

    def run(self: Self, time_steps: int, draw=False) -> None:
        for time_step in range(time_steps):
            if (time_step == FRAME_REBUTTAL_STARTS):
                infected_nodes = self.pick_random_ignorant_nodes(INITIAL_TRUTH_ADVOCATES)
                for node in infected_nodes:
                    self.states[node] = State.TRUTH_ADVOCATE
            if draw:
                self.draw()
            self.step()
        if draw:
            self.draw()

# Create a graph
POPULATION_SIZE = 4039
CLIQUENESS = 44
graph = nx.barabasi_albert_graph(POPULATION_SIZE, CLIQUENESS)

# Status colours
STATUS_COLOURS = {
    State.IGNORANT: 'blue',
    State.IGNORANT_REMOVAL: 'darkblue',
    State.RUMOUR_ADVOCATE: 'red',
    State.RUMOUR_SPREADER: 'orange',
    State.RUMOUR_CARRIER: 'gold',
    State.TRUTH_CARRIER: 'lawngreen',
    State.TRUTH_SPREADER: 'limegreen',
    State.TRUTH_ADVOCATE: 'darkgreen'
}

# Parameters for the DK model
INITIAL_RUMOUR_ADVOCATES = 1
FRAME_REBUTTAL_STARTS = 5
INITIAL_TRUTH_ADVOCATES = 5
TIME_STEPS = 50

TRUTH_ACCEPTANCE_PROB = 0.5
RUMOUR_ACCEPTANCE_PROB = 0.20

RUMOUR_CARRIER_TO_SPREADER_PROB = 0.05
TRUTH_CARRIER_TO_SPREADER_PROB = 0.2

RUMOUR_SPREADER_TO_ADVOCATE_PROB = 0.1
TRUTH_SPREADER_TO_ADVOCATE_PROB = 0.2

RUMOUR_SPREADER_TO_TRUTH_PROB = 0.01
TRUTH_SPREADER_TO_RUMOUR_PROB = 0.005

RUMOUR_CARRIER_TO_TRUTH_PROB = 0.1
TRUTH_CARRIER_TO_RUMOUR_PROB = 0.05

IGNORE_PROB = 0.1

# Run the ICSAR model
dk_model = ICSARModel(graph)
dk_model.run(TIME_STEPS, draw=False)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(dk_model.i_history, label='Ignorant', color=STATUS_COLOURS[State.IGNORANT])
plt.plot(dk_model.ir_history, label='Ignorant Removal', color=STATUS_COLOURS[State.IGNORANT_REMOVAL])
plt.plot(dk_model.ra_history, label='Rumour Advocate', color=STATUS_COLOURS[State.RUMOUR_ADVOCATE])
plt.plot(dk_model.rs_history, label='Rumour Spreader', color=STATUS_COLOURS[State.RUMOUR_SPREADER])
plt.plot(dk_model.rc_history, label='Rumour Carrier', color=STATUS_COLOURS[State.RUMOUR_CARRIER])
plt.plot(dk_model.tc_history, label='Truth Carrier', color=STATUS_COLOURS[State.TRUTH_CARRIER])
plt.plot(dk_model.ts_history, label='Truth Spreader', color=STATUS_COLOURS[State.TRUTH_SPREADER])
plt.plot(dk_model.ta_history, label='Truth Advocate', color=STATUS_COLOURS[State.TRUTH_ADVOCATE])
plt.title('DK Model of Rumor Propagation on a Graph')
plt.xlabel('Time Steps')
plt.ylabel('Number of Individuals')
plt.legend()
plt.show()

# Running 100 simulations to get the final rumour sizes
# final_rumour_sizes = []

# for _ in range(100):
#     dk_model = ICSARModel(graph)
#     dk_model.run(TIME_STEPS, draw=False)
#     final_rumour_sizes.append(dk_model.ta_history[-1] / POPULATION_SIZE)

# plt.figure(figsize=(10, 6))
# plt.hist(final_rumour_sizes, bins=20, range=(0, 1))
# plt.title('Distribution of Final Rumour Sizes')
# plt.xlabel('Final Rumour Size')
# plt.ylabel('Frequency')
# plt.show()
