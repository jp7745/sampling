#!/usr/bin/env python3



# Copyright 2023 L3Harris Technologies, Inc.
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#

import uuid


import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



def convert_index_to_coordinates(i: int, dimensions: np.ndarray) -> np.ndarray:
    """_summary_

    Args:
        i (int): _description_
        dimensions (np.ndarray): _description_

    Returns:
        np.ndarray: _description_
    """
    remainder = i
    coords = np.zeros(len(dimensions))
    for d in range(len(dimensions)):
        coords[d] = int(remainder // np.product(dimensions[d+1:]))
        remainder = int(remainder % np.product(dimensions[d+1:]))
    return coords


def convert_coordinates_to_index(coords: np.ndarray, dimensions: np.ndarray) -> int:
    """_summary_

    Args:
        coords (np.ndarray): _description_
        dimensions (np.ndarray): _description_

    Returns:
        int: _description_
    """
    i = 0
    for d in range(len(coords)):
        i += coords[d]*np.product(dimensions[d+1:])
    return i


def convert_spin_config_to_state_index(spin_config: np.ndarray) -> int:
    """_summary_

    Args:
        spin_config (np.ndarray): _description_

    Returns:
        int: _description_
    """
    state = 0x0
    for i in range(len(spin_config)):
        if spin_config[i] == 1:
            state = state | (0x1 << i)
    return state

def convert_state_index_to_spin_config(n: int, state_index: int) -> np.ndarray:
    """_summary_

    Args:
        n (int): _description_
        state_index (int): _description_

    Returns:
        np.ndarray: _description_
    """
    spin_config = np.zeros(n)
    for i in range(n):
        spin_config[i] = 1 - 2*bool(state_index & (0x1<<i))
    return spin_config


def generate_random_spin_config(n: int) -> np.ndarray:
    """_summary_

    Args:
        n (int): _description_

    Returns:
        np.ndarray: _description_
    """
    return 2*(np.random.random(n) > 0.5*np.ones(n)) - np.ones(n)





def init_square_grid_nodes(dimensions: list=None) -> dict:
    """_summary_

    Args:
        dimensions (list, optional): _description_. Defaults to None.

    Returns:
        dict: _description_
    """
    nodes = {}
    n = np.product(dimensions)
    for i in range(n):
        nodes[i] = {
            "position":list((convert_index_to_coordinates(i, dimensions))),
            "spin":1.0, # default is all +1.
            "B":0.0 # default is no external field
        }
    return nodes







def init_square_grid_internal_edges(
        nodes: dict=None,
        distance_threshold: float=1.01
    ) -> dict:
    """_summary_

    Args:
        nodes (dict, optional): _description_. Defaults to None.
        distance_threshold (float, optional): _description_. Defaults to 1.01.

    Returns:
        dict: _description_
    """
    edges = {}
    for i in nodes:
        for j in nodes:
            if i < j:
                xyz_i = np.array(nodes[i]["position"])
                xyz_j = np.array(nodes[j]["position"])
                distance = np.linalg.norm(xyz_i - xyz_j)
                if distance < distance_threshold:
                    edges[(i,j)] = {
                        "weight":1.0, # default is 1.0
                        "periodic_boundary_edge":False  # internal edge
                    }
    return edges

def add_square_grid_periodic_boundary_edges(
        dimensions: list=None,
        nodes: dict=None,
        edges: dict=None
    ) -> dict:
    """_summary_

    Args:
        dimensions (list, optional): _description_. Defaults to None.
        nodes (dict, optional): _description_. Defaults to None.
        edges (dict, optional): _description_. Defaults to None.

    Returns:
        dict: _description_
    """


    for i in nodes:
        i_x = nodes[i]["position"][0]
        i_y = nodes[i]["position"][1]
        i_z = nodes[i]["position"][2]
        if i_x == 0: 
            for j in nodes:
                if i < j:
                    if nodes[j]["position"][0] == dimensions[0] - 1:
                        if nodes[j]["position"][1] == i_y:
                            if nodes[j]["position"][2] == i_z:
                                if (i, j) not in edges:
                                    edges[(i,j)]={
                                        "weight":1.0, # init to 1
                                        "periodic_boundary_edge":True
                                    }
                                    break
        if i_x == dimensions[0] - 1: 
            for j in nodes:
                if i < j:
                    if nodes[j]["position"][0] == 0:
                        if nodes[j]["position"][1] == i_y:
                            if nodes[j]["position"][2] == i_z:
                                if (i, j) not in edges:
                                    edges[(i,j)]={
                                        "weight":1.0, # init to 1
                                        "periodic_boundary_edge":True
                                    }
                                    break
        if i_y == 0:
            for j in nodes:
                if i < j:
                    if nodes[j]["position"][0] == i_x:
                        if nodes[j]["position"][1] == dimensions[1] - 1:
                            if nodes[j]["position"][2] == i_z:
                                if (i, j) not in edges:
                                    edges[(i,j)]={
                                        "weight":1.0, # init to 1
                                        "periodic_boundary_edge":True
                                    }
                                    break
        if i_y == dimensions[1] - 1:
            for j in nodes:
                if i < j:
                    if nodes[j]["position"][0] == i_x:
                        if nodes[j]["position"][1] == 0:
                            if nodes[j]["position"][2] == i_z:
                                if (i, j) not in edges:
                                    edges[(i,j)]={
                                        "weight":1.0, # init to 1
                                        "periodic_boundary_edge":True
                                    }
                                    break
        if i_z == 0:
            for j in nodes:
                if i < j:
                    if nodes[j]["position"][0] == i_x:
                        if nodes[j]["position"][1] == i_y:
                            if nodes[j]["position"][2] == i_z:
                                if (i, j) not in edges:
                                    edges[(i,j)]={
                                        "weight":1.0, # init to 1
                                        "periodic_boundary_edge":True
                                    }
                                    break
        if i_z == dimensions[2] - 1:
            for j in nodes:
                if i < j:
                    if nodes[j]["position"][0] == i_x:
                        if nodes[j]["position"][1] == i_y:
                            if nodes[j]["position"][2] == i_z:
                                if (i, j) not in edges:
                                    edges[(i,j)]={
                                        "weight":1.0, # init to 1
                                        "periodic_boundary_edge":True
                                    }
                                    break
    return edges







class Graph(nx.Graph):
    """Extends the networkX Graph class
    DESCRIPTION: _summary_
    """

    def __init__(
            self, 
            nx_graph: nx.Graph=None,
            nodes: dict=None,
            edges: dict=None,
            name: str="default_name",
            user_defined_uuid: str=None
        ):
        """_summary_

        Args:
            nx_graph (nx.Graph, optional): _description_. Defaults to None.
            nodes (dict, optional): _description_. Defaults to None.
            edges (dict, optional): _description_. Defaults to None.
            name (str, optional): _description_. Defaults to "default_name".
            uuid (str, optional): _description_. Defaults to None.
        """
        # initialize an empty NetworkX graph:
        super(Graph, self).__init__(nx_graph)
        self.load_nodes(nodes)
        self.load_edges(edges)
        self.name = name
        if user_defined_uuid is None:
            self.uuid = str(uuid.uuid4())
        else:
            self.uuid = user_defined_uuid



    def load_nodes(self, nodes: dict=None) -> None:
        """_summary_

        Args:
            nodes (dict, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if nodes == None:
            return None
        else:
            for i in nodes:
                self.add_node(
                    i,
                    position=nodes[i]["position"],
                    spin=nodes[i]["spin"],
                    B=nodes[i]["B"]
                )
            return None


    def load_edges(self, edges: dict=None) -> None:
        """_summary_

        Args:
            edges (dict, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if edges == None:
            return None
        else:
            for e in edges:
                i = e[0]
                j = e[1]
                self.add_edge(
                    i,
                    j,
                    weight=edges[e]["weight"],
                    periodic_boundary_edge=edges[e]["periodic_boundary_edge"]
                )
            return None


    
    def plot(self) -> None:
        """A Matplotlib figure is rendered.
        """

        node_xyz = np.array([self.nodes[i]["position"] for i in self.nodes()])
        edge_xyz = np.array([(self.nodes[e[0]]["position"], self.nodes[e[1]]["position"]) for e in self.edges()])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("$x$ position")
        ax.set_ylabel("$y$ position")
        ax.set_zlabel("$z$ position")
        ax.set_title("System of Interest")
        colors = np.array([self.nodes[i]["spin"] for i in self.nodes()])
        ax.scatter(
            *node_xyz.T, 
            c=colors,
            s=100, 
            ec="w",
            vmin=-1.0,
            vmax=1.0
        )
        for vizedge in edge_xyz:
            ax.plot(*vizedge.T, color="tab:gray")
        fig.tight_layout()
        plt.show()


    def set_spins(self, spin_config: np.ndarray) -> None:
        """_summary_

        Args:
            spin_config (np.ndarray): _description_
        """
        for i in self.nodes():
            self.nodes[i]["spin"] = spin_config[i]

    def get_spins(self) -> np.ndarray:
        """_summary_

        Returns:
            np.ndarray: _description_
        """
        return np.array([self.nodes[i]["spin"] for i in self.nodes()])
        

    def set_external_field(self, B: np.ndarray) -> None:
        """_summary_

        Args:
            B (np.ndarray): _description_
        """
        for i in self.nodes():
            self.nodes[i]["B"] = B[i]

    def get_external_field(self) -> np.ndarray:
        """_summary_

        Returns:
            np.ndarray: _description_
        """
        return np.array([self.nodes[i]["B"] for i in self.nodes()])


    def set_couplings(self, J: dict) -> None:
        """_summary_

        Args:
            J (dict): _description_
        """
        if len(self.edges()) == 0:
            raise Exception("Error: no edges in the graph!")
        for e in self.edges():
            i = e[0]
            j = e[1]
            self.edges[e]["weight"] = J[i, j]

    def get_couplings(self) -> dict:
        """_summary_

        Returns:
            dict: _description_
        """
        return {e:self.edges[e]["weight"] for e in self.edges()}

    def energy(self) -> float :
        """$H(\sigma) = - \sum_{i~j} J_{ij} \sigma_i \sigma_j - \sum_i B_i \sigma_i$
        Note that we are going with the classic definition with the minus signs.

        Note that we allow spin for each node to be +1, -1, but in our "constructing" starting spin_configs we allow spin to be 0 for unset nodes.

        Returns:
            float: The energy of the system as a real number.
        """
        
        energy_sum = 0.0
        for i in self.nodes():
            energy_sum -= self.nodes[i]["B"]*self.nodes[i]["spin"]
        for e in self.edges():
            i = e[0]
            j = e[1]
            energy_sum -= self.edges[(i, j)]["weight"] * \
                self.nodes[i]["spin"]*self.nodes[j]["spin"]
        return energy_sum

    def mcmc_sample_v1(self,
            num_steps: int,
            beta: float,
            rng_seed: int=None,
            print_status: bool=None
        ) -> np.ndarray :
        """
        TODO
        """
        
        if rng_seed is not None:
            raise Exception("Error: fixed RNG seed not implemented!")

        
        n = len(self.nodes())
        i = 0
        for step in range(num_steps):
            self.mcmc_mh_step(
                beta=beta,
                target_spin=i
            )
            i = i % n

            if print_status:
                if step % 50 == 0:
                    print("step:",step,"/",num_steps)
        
        return self.energy(), self.get_spins()



    def mcmc_mh_step(self,
            beta: float,
            target_spin: int=None,
            uniform_random_number: float=None,
        ) -> float :
        """
        TODO
        """

        if uniform_random_number is None:
            uniform_random_number = np.random.random()

        current_energy = self.energy()
        self.nodes[target_spin]["spin"] *= -1 # flip sign
        proposed_energy = self.energy()

        if proposed_energy < (current_energy - 1e-9):
            # accept proposed state because it is lower energy.
            current_energy = proposed_energy
        else:
            threshold = np.exp(-beta*(proposed_energy - current_energy))
            if uniform_random_number < threshold:
                # accept proposed state even though it is higher energy
                current_energy = proposed_energy
            else:
                # do NOT accept proposed state
                # keep E_current set as is (no update)
                self.nodes[target_spin]["spin"] *= -1 # revert to keep the "current" state                

        return current_energy, self.get_spins()       








################################################################################
# Functions used to generate greedy randomized spin configurations

def is_complete_spin_config(spin_config: np.ndarray) -> bool:
    """_summary_

    Args:
        spin_config (np.ndarray): _description_

    Returns:
        bool: _description_
    """

    n = len(spin_config)
    return bool(np.prod(abs(spin_config) > 1e-6*np.ones(n)))




def generate_randomized_greedy_spin_config(
        G: Graph=None,
        beta: float=None,
        plot_progress: bool=False
    ) -> np.ndarray:
    """_summary_

    Args:
        G (sampling.Graph, optional): _description_. Defaults to None.
        beta (float, optional): _description_. Defaults to None.
        plot_progress (bool, optional): _description_. Defaults to False.

    Returns:
        np.ndarray: _description_
    """

    n = len(G.nodes())
    spin_config = np.zeros(n) # an empty config
    i = np.random.randint(0,n-1) # randomly pick first node i
    spin_config[i] = 2*(np.random.random() > 0.5) - 1.0 # randomly set node i to +1.0 or -1.0.
    G.set_spins(spin_config)
            
    if plot_progress:
        G.plot()
        print("spin config:",G.get_spins())
        print("Energy:",G.energy())

    while not is_complete_spin_config(spin_config):
        
        nbors, conn_nbors = collect_set_of_fixed_spin_neighbors(
            G=G, 
            spin_config=spin_config
        )

        #randomly pick a candidate neighbor
        # TODO:  weighted random pick by conn_nbors
        if len(nbors) == 1:
            j = nbors[0]
        else:
            nbor_index = np.random.randint(0,len(nbors)-1) 
            j = nbors[nbor_index]

        candidate_spin_config_down = spin_config.copy()
        candidate_spin_config_down[j] = -1.0

        candidate_spin_config_up = spin_config.copy()
        candidate_spin_config_up[j] = 1.0

        G.set_spins(candidate_spin_config_down)
        E_candidate_down = G.energy()
        e_down = np.exp(-1.0*beta*E_candidate_down)

        G.set_spins(candidate_spin_config_up)
        E_candidate_up = G.energy()
        e_up = np.exp(-1.0*beta*E_candidate_up)

        prob_down = e_down / (e_down + e_up)

        u = np.random.rand()
        if u < prob_down:
            spin_config[j] = -1.0
        else:
            spin_config[j] = 1.0

        G.set_spins(spin_config)
            
        if plot_progress:
            G.plot()
            print("spin config:",G.get_spins())
            print("Energy:",G.energy())
        


    return spin_config










def collect_set_of_fixed_spin_neighbors(
        G: Graph=None, 
        spin_config: np.ndarray=None
    ) -> np.ndarray | np.ndarray:
    """_summary_

    Args:
        G (sampling.Graph, optional): _description_. Defaults to None.
        spin_config (np.ndarray, optional): _description_. Defaults to None.

    Returns:
        np.ndarray | np.ndarray: _description_
    """

    n = len(G.nodes())
    set_of_neighbors_with_no_spin_yet = []

    # assemble the set of nodes that are adjacent to nodes with spin already assigned.
    for i in range(n):
        if abs(spin_config[i]) > 1e-6: #nonzero
            for j in G.neighbors(i):
                if j not in set_of_neighbors_with_no_spin_yet:
                    if abs(spin_config[j]) < 1e-6: #close to zero (spin not set)
                        set_of_neighbors_with_no_spin_yet.append(j)

    #convert list to NumPy Array.
    set_of_neighbors_with_no_spin_yet = np.array(
        set_of_neighbors_with_no_spin_yet) 
    
    neighbor_connectivity_to_fixed_spins = np.zeros(
        len(set_of_neighbors_with_no_spin_yet),dtype=np.int64)

    # count the number of edges connecting each neighbor node to nodes with spin already assigned
    for j_index in range(len(set_of_neighbors_with_no_spin_yet)):
        j = set_of_neighbors_with_no_spin_yet[j_index]
        for i in G.neighbors(j):
            if abs(spin_config[i]) > 1e-6: #nonzero
                neighbor_connectivity_to_fixed_spins[j_index] += 1
    

    # sort lists before returning
    index_sort = np.argsort(neighbor_connectivity_to_fixed_spins)
    neighbor_connectivity_to_fixed_spins = neighbor_connectivity_to_fixed_spins[index_sort]
    set_of_neighbors_with_no_spin_yet = set_of_neighbors_with_no_spin_yet[index_sort]

    return set_of_neighbors_with_no_spin_yet, neighbor_connectivity_to_fixed_spins


