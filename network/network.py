from typing import Self
from networkx import Graph
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from enum import Enum

class State(Enum):
    SPREADER = 0
    IGNORANT = 1
    STIFLER = 2

class DKModel:
    def __init__(self: Self, graph: Graph, initial_infected: int) -> None:
        self.graph = graph
        
        # Initialize states
        self.states = {node: State.IGNORANT for node in graph.nodes}
        infected_nodes = np.random.choice(graph.nodes, initial_infected, replace=False)
        for node in infected_nodes:
            self.states[node] = State.SPREADER
        
        # History tracking
        self.s_history = []
        self.i_history = []
        self.r_history = []

    def step(self: Self) -> None:
        new_states = self.states.copy()
        
        for node in self.graph.nodes:
            if self.states[node] == State.SPREADER:
                neighbors = list(self.graph.neighbors(node))

                for neighbor in neighbors:
                    # If neighbour is ignorant
                    if self.states[neighbor] == State.IGNORANT:
                        if np.random.random() < RUMOUR_ACCEPTANCE_PROB:
                            new_states[neighbor] = State.SPREADER
                        elif np.random.random() < 1 - RUMOUR_ACCEPTANCE_PROB:
                            new_states[neighbor] = State.STIFLER
                    # If neighbour is a spreader
                    elif self.states[neighbor] == State.SPREADER:
                        if np.random.random() < SPREADER_SPREADER_STIFLE_PROB:
                            new_states[neighbor] = State.STIFLER
                    # If neighbour is a stifler
                    elif self.states[neighbor] == State.STIFLER:
                        if np.random.random() < SPREADER_STIFLER_STIFLE_PROB:
                            new_states[neighbor] = State.STIFLER
                
                # Random chance to become a stifler
                if np.random.random() < FORGET_PROB:
                    new_states[node] = State.STIFLER
                    continue

                # If all neighbours are stiflers, the spreader becomes a stifler
                if all(self.states[neighbor] == State.STIFLER for neighbor in neighbors):
                    new_states[node] = State.STIFLER
                    continue
        
        self.states = new_states
        
        # Count states
        spreaders = list(self.states.values()).count(State.SPREADER)
        ignorants = list(self.states.values()).count(State.IGNORANT)
        stiflers = list(self.states.values()).count(State.STIFLER)

        self.s_history.append(spreaders)
        self.i_history.append(ignorants)
        self.r_history.append(stiflers)

    def draw(self: Self) -> None:
        plt.figure(figsize=(10, 6))

        # Assemble colour map
        colours = []
        for node in self.graph.nodes:
            if self.states[node] == State.SPREADER:
                colours.append('red')
            elif self.states[node] == State.IGNORANT:
                colours.append('blue')
            elif self.states[node] == State.STIFLER:
                colours.append('green')

        nx.draw(self.graph, node_size=50, node_color=colours, with_labels=False)
        plt.title('DK Model of Rumor Propagation on a Graph')
        plt.show()

    def run(self: Self, time_steps: int, draw=False) -> None:
        for _ in range(time_steps):
            if draw:
                self.draw()
            self.step()

            # If all nodes are either spreaders or stiflers, stop the simulation
            if all(state == State.STIFLER or state == State.SPREADER for state in self.states.values()):
                break
        if draw:
            self.draw()

# Create a graph
population_size = 4039
graph = nx.barabasi_albert_graph(population_size, 44)

# Parameters for the DK model
RUMOUR_ACCEPTANCE_PROB = 0.20
SPREADER_SPREADER_STIFLE_PROB = 0.05
SPREADER_STIFLER_STIFLE_PROB = 0.1
FORGET_PROB = 0.5

# Run the DK model
dk_model = DKModel(graph, initial_infected=1)
dk_model.run(100, draw=False)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(dk_model.s_history, label='Spreader')
plt.plot(dk_model.i_history, label='Ignorant')
plt.plot(dk_model.r_history, label='Stifler')
plt.title('DK Model of Rumor Propagation on a Graph')
plt.xlabel('Time Steps')
plt.ylabel('Number of Individuals')
plt.legend()
plt.show()

# Running 100 simulations to get the final rumour sizes
# final_rumour_sizes = []

# for _ in range(100):
#     dk_model = DKModel(graph, initial_infected=1)
#     dk_model.run(100, draw=False)
#     final_rumour_sizes.append(dk_model.s_history[-1] / population_size)

# plt.figure(figsize=(10, 6))
# plt.hist(final_rumour_sizes, bins=20, range=(0, 1))
# plt.title('Distribution of Final Rumour Sizes')
# plt.xlabel('Final Rumour Size')
# plt.ylabel('Frequency')
# plt.show()
