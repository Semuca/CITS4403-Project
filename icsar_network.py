from typing import Self
import numpy as np
import matplotlib.pyplot as plt
from graph_tool.all import Graph, graph_draw, sfdp_layout, closeness
import networkx as nx

from enum import Enum

def convert_nx_graph_to_gt(nx_graph):
    gt_graph = Graph(directed=False)
    gt_graph.add_edge_list(nx_graph.edges)
    return gt_graph

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
        self.states = self.graph.new_vp("string")
        self.colours = graph.new_vp("string")
        for v in self.graph.vertices():
            self.states[v] = State.IGNORANT.name

        initial_rumour_advocates = self.pick_random_ignorant_nodes(INITIAL_RUMOUR_ADVOCATES)
        for v in initial_rumour_advocates:
            self.states[v] = State.RUMOUR_ADVOCATE.name
        
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
        vertices = list(self.graph.vertices())
        ignorant_nodes = [vertices[index] for index, state in enumerate(self.states) if state == State.IGNORANT.name or state == State.IGNORANT_REMOVAL.name]
        return np.random.choice(ignorant_nodes, number, replace=False)

    def step(self: Self) -> None:
        new_states = self.states.copy()
        
        for node in self.graph.vertices():
            state = self.states[node]
            if state == State.RUMOUR_SPREADER.name or state == State.RUMOUR_ADVOCATE.name:
                neighbors = list(self.graph.get_all_neighbours(node))

                if state == State.RUMOUR_SPREADER.name and np.random.random() < RUMOUR_SPREADER_TO_ADVOCATE_PROB:
                    new_states[node] = State.RUMOUR_ADVOCATE.name

                for neighbor in neighbors:
                    # If neighbour is ignorant
                    if self.states[neighbor] == State.IGNORANT.name:
                        if np.random.random() < RUMOUR_ACCEPTANCE_PROB:
                            new_states[neighbor] = State.RUMOUR_CARRIER.name
                        elif np.random.random() < IGNORE_PROB:
                            new_states[neighbor] = State.IGNORANT_REMOVAL.name
                    # If neighbour is rumour carrier
                    elif self.states[neighbor] == State.RUMOUR_CARRIER.name:
                        if np.random.random() < RUMOUR_CARRIER_TO_SPREADER_PROB:
                            new_states[neighbor] = State.RUMOUR_SPREADER.name
                    # If neighbour is truth carrier
                    elif self.states[neighbor] == State.TRUTH_CARRIER.name:
                        if np.random.random() < TRUTH_CARRIER_TO_RUMOUR_PROB:
                            new_states[neighbor] = State.RUMOUR_CARRIER.name
                    # If neighbour is truth spreader
                    elif self.states[neighbor] == State.TRUTH_SPREADER.name:
                        if np.random.random() < TRUTH_SPREADER_TO_RUMOUR_PROB:
                            new_states[neighbor] = State.RUMOUR_CARRIER.name
            elif state == State.TRUTH_SPREADER.name or state == State.TRUTH_ADVOCATE.name:
                neighbors = list(self.graph.get_all_neighbours(node))

                if state == State.TRUTH_SPREADER.name and np.random.random() < TRUTH_SPREADER_TO_ADVOCATE_PROB:
                    new_states[node] = State.TRUTH_ADVOCATE.name

                for neighbor in neighbors:
                    # If neighbour is ignorant
                    if self.states[neighbor] == State.IGNORANT.name:
                        if np.random.random() < TRUTH_ACCEPTANCE_PROB:
                            new_states[neighbor] = State.TRUTH_CARRIER.name
                        elif np.random.random() < IGNORE_PROB:
                            new_states[neighbor] = State.IGNORANT_REMOVAL.name
                    # If neighbour is truth carrier
                    elif self.states[neighbor] == State.TRUTH_CARRIER.name:
                        if np.random.random() < TRUTH_CARRIER_TO_SPREADER_PROB:
                            new_states[neighbor] = State.TRUTH_SPREADER.name
                    # If neighbour is rumour carrier
                    elif self.states[neighbor] == State.RUMOUR_CARRIER.name:
                        if np.random.random() < RUMOUR_CARRIER_TO_TRUTH_PROB:
                            new_states[neighbor] = State.TRUTH_CARRIER.name
                    # If neighbour is rumour spreader
                    elif self.states[neighbor] == State.RUMOUR_SPREADER.name:
                        if np.random.random() < RUMOUR_SPREADER_TO_TRUTH_PROB:
                            new_states[neighbor] = State.TRUTH_CARRIER.name
        
        self.states = new_states
        
        # Count states
        self.i_history.append(list(self.states).count(State.IGNORANT.name))
        self.ir_history.append(list(self.states).count(State.IGNORANT_REMOVAL.name))
        self.ra_history.append(list(self.states).count(State.RUMOUR_ADVOCATE.name))
        self.rs_history.append(list(self.states).count(State.RUMOUR_SPREADER.name))
        self.rc_history.append(list(self.states).count(State.RUMOUR_CARRIER.name))
        self.tc_history.append(list(self.states).count(State.TRUTH_CARRIER.name))
        self.ts_history.append(list(self.states).count(State.TRUTH_SPREADER.name))
        self.ta_history.append(list(self.states).count(State.TRUTH_ADVOCATE.name))

    def draw(self: Self) -> None:
        # Assemble colour map
        for node in self.graph.vertices():
            state = self.states[node]
            self.colours[node] = (STATUS_COLOURS[state])

        graph_draw(self.graph, pos=sfdp_layout(self.graph), node_size=50, vertex_fill_color=self.colours, with_labels=False, output="dk_model.svg", vorder=closeness(self.graph))

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
# POPULATION_SIZE = 4039
# CLIQUENESS = 44
# nx_graph = nx.barabasi_albert_graph(POPULATION_SIZE, CLIQUENESS)
# graph = convert_nx_graph_to_gt(nx_graph)

graph = Graph()

with open("datasets/facebook_combined.txt","r") as f:
    edges = np.fromiter((int(n) for x in f.readlines() for n in x.split()), dtype='int')

unique_vertices = np.unique(edges)
num_vertices = len(unique_vertices)
vertex_mapping = {vertex: i for i, vertex in enumerate(unique_vertices)}
edges = np.fromiter((vertex_mapping[vertex] for vertex in edges), dtype='int')
edges = edges.reshape((-1, 2))
del unique_vertices
del vertex_mapping

# Build graph
graph = Graph(directed=False)
graph.add_vertex(num_vertices)
graph.add_edge_list(edges)

# Status colours
STATUS_COLOURS = {
    State.IGNORANT.name: 'blue',
    State.IGNORANT_REMOVAL.name: 'darkblue',
    State.RUMOUR_ADVOCATE.name: 'red',
    State.RUMOUR_SPREADER.name: 'orange',
    State.RUMOUR_CARRIER.name: 'gold',
    State.TRUTH_CARRIER.name: 'lawngreen',
    State.TRUTH_SPREADER.name: 'limegreen',
    State.TRUTH_ADVOCATE.name: 'darkgreen'
}

# Parameters for the DK model
INITIAL_RUMOUR_ADVOCATES = 1
FRAME_REBUTTAL_STARTS = 5
INITIAL_TRUTH_ADVOCATES = 5
TIME_STEPS = 3

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
dk_model.run(TIME_STEPS, draw=True)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(dk_model.i_history, label='Ignorant', color=STATUS_COLOURS[State.IGNORANT.name])
plt.plot(dk_model.ir_history, label='Ignorant Removal', color=STATUS_COLOURS[State.IGNORANT_REMOVAL.name])
plt.plot(dk_model.ra_history, label='Rumour Advocate', color=STATUS_COLOURS[State.RUMOUR_ADVOCATE.name])
plt.plot(dk_model.rs_history, label='Rumour Spreader', color=STATUS_COLOURS[State.RUMOUR_SPREADER.name])
plt.plot(dk_model.rc_history, label='Rumour Carrier', color=STATUS_COLOURS[State.RUMOUR_CARRIER.name])
plt.plot(dk_model.tc_history, label='Truth Carrier', color=STATUS_COLOURS[State.TRUTH_CARRIER.name])
plt.plot(dk_model.ts_history, label='Truth Spreader', color=STATUS_COLOURS[State.TRUTH_SPREADER.name])
plt.plot(dk_model.ta_history, label='Truth Advocate', color=STATUS_COLOURS[State.TRUTH_ADVOCATE.name])
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
