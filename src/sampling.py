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
import json

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
        spin_config[i] = -1.0 + 2*bool(state_index & (0x1<<i))
    return spin_config


def generate_random_spin_config(n: int) -> np.ndarray:
    """_summary_

    Args:
        n (int): _description_

    Returns:
        np.ndarray: _description_
    """
    return 2*(np.random.random(n) > 0.5*np.ones(n)) - np.ones(n)



def load_graph_instance_from_file(filename: str=None):
    """_summary_

    Args:
        filename (str, optional): _description_. Defaults to None.

    Returns:
        Graph: a sampling.Graph object.
    """
    with open(filename,"rb") as f:
        read_bytes = f.read()
        f.close()

    # convert the byte array to a dict
    instance = json.loads(read_bytes)
    
    G_nx = nx.node_link_graph(instance["graph_data"])
    G = Graph(G_nx)
    
    G.name = instance["metadata"]["graph_name"]
    G.uuid = instance["metadata"]["graph_uuid"]
    G.temperature_T = instance["metadata"]["temperature_T"]
    G.k_B = instance["metadata"]["k_B"]
    
    G.synchronize_attributes()
    
    return G
    



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
        
        self.name = name
        self.temperature_T = None
        self.k_B = None
        
        if user_defined_uuid is None:
            self.uuid = str(uuid.uuid4())
        else:
            self.uuid = user_defined_uuid

        # after other parameters are updated, 
        # the following may be updated by calling the .synchronize_attributes() method.
        self._has_external_field = None
        self._consistent_external_field = None
        self._edge_type = None # antiferromagnetic, ferromagnetic, spin-glass
        self._consistent_edge_weights = None
        self._beta = None
        self._has_periodic_boundary_edges = None

        # _exact_probability_distribution is updated by calling the
        # .brute_force_probability_distribution_calculation() method.
        self._exact_probability_distribution = None

        if nodes is not None:
            self.load_nodes(nodes)
        if edges is not None:
            self.load_edges(edges)
        


        


    def load_nodes(self, nodes: dict=None):
        """_summary_

        Args:
            nodes (dict, optional): _description_. Defaults to None.

        """
        for i in nodes:
            self.add_node(
                i,
                position=nodes[i]["position"],
                spin=nodes[i]["spin"],
                B=nodes[i]["B"]
            )
        self.synchronize_attributes()


    def load_edges(self, edges: dict=None):
        """
        xxxxxxxxxx
        """

        for e in edges:
            i = e[0]
            j = e[1]
            self.add_edge(
                i,
                j,
                weight=edges[e]["weight"],
                periodic_boundary_edge=edges[e]["periodic_boundary_edge"]
            )
        self.synchronize_attributes()

    def write_instance_to_file(self,
            relative_path: str=None,
            instance_uuid: str=None,
            filename: str=None,
        ):
        """_summary_

        Args:
            relative_path (str, optional): _description_. Defaults to None.
            instance_uuid (str, optional): _description_. Defaults to None.
            filename (str, optional): _description_. Defaults to None.
        """
        self.synchronize_attributes()

        if instance_uuid is None:
            instance_uuid = str(uuid.uuid4())

        # we made the conscious decision to preserve and use the data structure
        # provided/read by the NetworkX.node_link_data() function.
        # other attributes appear in the metadata dictionary.
        # specifically exporting a graph using the .node_link_data() function
        # does NOT include .temperature_T and ._beta and some other attributes.
        graph_data = nx.node_link_data(self)

        benchmark_requirements = {
            "num_samples_k":1000,
            "time_limit_seconds":600
            }
        metadata = self.get_summary_details()
        metadata["instance_uuid"] = instance_uuid

        aggregated_dictionary = {
            "benchmark_requirements":benchmark_requirements,
            "metadata":metadata,
            "graph_data":graph_data
        }

        # write the dict as JSON to a file
        if filename is None:
            full_filename = relative_path + "/instance." + instance_uuid + ".json"
        else:
            full_filename = relative_path + "/" + filename
        
        output_bytes = json.dumps(aggregated_dictionary).encode("utf8")
        with open(full_filename,"wb") as f:
            f.write(output_bytes)
            f.close()

    
    def plot(self, show_node_lables: bool=False) -> None:
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
        
        if show_node_lables:
            for i in self.nodes():
                pos_i = self.nodes[i]["position"]
                ax.text(pos_i[0], pos_i[1], pos_i[2], str(i))
        fig.tight_layout()
        plt.show()

    def get_adjacency_matrix_with_zero_diagonal_as_numpy_matrix(self) -> np.ndarray:
        """_summary_

        Returns:
            np.ndarray: _description_
        """
        adj_matrix = nx.to_numpy_array(self)
        external_field_B = np.array([ self.nodes[i]["B"] for i in range(len(self.nodes()))])
        
        return adj_matrix, external_field_B

    def brute_force_probability_distribution_calculation(self):
        """Calculates (or updates) Graph._exact_probability_distribution in place.
        TODO
        """
        self.synchronize_attributes()

        n = len(self.nodes())
        N = 2**n
        self._exact_probability_distribution = np.zeros(N)
        for state_index in range(N):
            spin_config = convert_state_index_to_spin_config(n,state_index)
            self.set_spins(spin_config)
            E_state = self.get_energy()
            self._exact_probability_distribution[state_index] = np.exp(-1.0*self._beta*E_state)
            
        Z = sum(self._exact_probability_distribution) # calculate partition function
        self._exact_probability_distribution *= (1/Z) # normalize by partition function
    

    def synchronize_attributes(self):
        """
        This method updates and synchronizes all attributes of the sampling.Graph object
        that the user should not directly modify such as ._beta, ._edge_type.

        This method is typically called before/after a sampling.Graph object is 
        written/read from a file.

        This method should be called manually if a user edits any attributes.
        """
        if (self.temperature_T is not None) and (self.k_B is not None):
            self._beta = 1/(self.temperature_T*self.k_B)
        else:
            self._beta = None
        
        self._has_external_field = False
        for i in self.nodes():
            if np.abs(self.nodes[i]["B"]) > 1e-12:
                self._has_external_field = True
                break
        
        if len(self.nodes()) > 0:
            self._consistent_external_field = True
            val = self.nodes[0]["B"]
            for i in self.nodes():
                if np.abs(self.nodes[i]["B"] - val) > 1e-12:
                    self._consistent_external_field = False
                    break
        
        if len(self.edges()) > 0:
            self._consistent_edge_weights = True
            edge_1 = list(self.edges())[0]
            for edge_2 in self.edges():
                if edge_1 != edge_2:
                    w1 = self.edges[edge_1]["weight"]
                    w2 = self.edges[edge_2]["weight"]
                    if np.abs(w1 - w2) > 1e-12:
                        self._consistent_edge_weights = False
                        break
        
        self._has_periodic_boundary_edges = False
        for edge in self.edges():
            if self.edges[edge]["periodic_boundary_edge"]:
                self._has_periodic_boundary_edges = True


        # determine ._edge_type \in {ferromagnetic, antiferromagnetic, spin-glass}
        all_positive_weights = True
        for edge in self.edges():
            if self.edges[edge]["weight"] < -1e-12: 
                all_positive_weights = False
                break
        all_negative_weights = True
        for edge in self.edges():
            if self.edges[edge]["weight"] > 1e-12:
                all_negative_weights = False
        if all_positive_weights:
            self._edge_type = "ferromagnetic"
        if all_negative_weights:
            self._edge_type = "antiferromagnetic"
        if (not all_positive_weights) and (not all_negative_weights):
            self._edge_type = "spin-glass"

    def get_summary_details(self) -> dict:
        """
        Returns a dictionary of metadata about a sampling.Graph object.
        """
        self.synchronize_attributes()
        metadata = {}
        metadata["graph_name"] = self.name
        metadata["graph_uuid"] = self.uuid
        metadata["number_of_nodes"] = len(self.nodes())
        metadata["number_of_edges"] = len(self.edges())
        metadata["temperature_T"] = self.temperature_T
        metadata["k_B"] = self.k_B
        metadata["beta"] = self._beta
        metadata["edge_type"] = self._edge_type
        metadata["consistent_edge_weights"] = self._consistent_edge_weights
        if self._consistent_edge_weights:
            e = list(self.edges())[0]            
            metadata["edge_weight"] = self.edges[e]["weight"]
        else:
            metadata["edge_weight"] = "varies"
        metadata["has_periodic_boundary_edges"] = self._has_periodic_boundary_edges
        metadata["has_nonzero_external_field"] = self._has_external_field
        metadata["consistent_external_field"] = self._consistent_external_field
        if self._consistent_external_field:
            i = list(self.nodes())[0]
            metadata["external_field"] = self.nodes[i]["B"]
        else:
            metadata["external_field"] = "varies"
        return metadata

    def set_spins(self, spin_config: np.ndarray):
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
        

    def set_external_field(self, B: np.ndarray):
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


    def set_couplings(self, J: dict):
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

    def get_energy(self) -> float :
        """$H(\sigma) = - \sum_{i~j} J_{ij} \sigma_i \sigma_j - \sum_i B_i \sigma_i$
        Note that we are going with the classic definition with the minus signs.

        Note that we allow spin for each node to be +1, -1, but in our "constructing" starting spin_configs we allow spin to be 0 for unset nodes.

        J_ij > 0 implies ferromagnetic: neighbors prefer to be the same spin.

        J_ij < 0 implies antiferromagnetic: neighbors prefer to be opposite spins.

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

    def generate_mcmc_sample_v1(self,
            num_steps: int,
            rng_seed: int=None,
            print_status: bool=None
        ) -> np.ndarray :
        """
        This will generate a MCMC sample from the graph.  
        The Markov chain will start at whatever state the current spin config is.
        
        In addition to returning the energy and spin configuration of the sample, 
        the spin configuration of the graph will be updated in place and can be read later.

        """
        
        if rng_seed is not None:
            raise Exception("Error: fixed RNG seed not implemented!")

        
        n = len(self.nodes())
        i = 0
        for step in range(num_steps):
            self.generate_mcmc_mh_step(
                target_spin=i
            )
            i += 1
            i = i % n

            if print_status:
                if step % 50 == 0:
                    print("step:",step,"/",num_steps)
        
        return self.get_energy(), self.get_spins()



    def generate_mcmc_mh_step(self,
            target_spin: int=None,
            uniform_random_number: float=None,
        ) -> float :
        """
        TODO
        """

        if uniform_random_number is None:
            uniform_random_number = np.random.random()

        current_energy = self.get_energy()
        self.nodes[target_spin]["spin"] *= -1 # flip sign
        proposed_energy = self.get_energy()

        if proposed_energy < (current_energy - 1e-9):
            # accept proposed state because it is lower energy.
            current_energy = proposed_energy
        else:
            threshold = np.exp(-1.0*self._beta*(proposed_energy - current_energy))
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
        plot_progress: bool=False
    ) -> np.ndarray:
    """_summary_

    Args:
        G (sampling.Graph, optional): _description_. Defaults to None.
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
        print("Energy:",G.get_energy())

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
        E_candidate_down = G.get_energy()
        e_down = np.exp(-1.0*G._beta*E_candidate_down)

        G.set_spins(candidate_spin_config_up)
        E_candidate_up = G.get_energy()
        e_up = np.exp(-1.0*G._beta*E_candidate_up)

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
            print("Energy:",G.get_energy())
        


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


