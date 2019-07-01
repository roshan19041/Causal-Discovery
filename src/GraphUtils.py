#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  9 14:10:14 2019

@author: roshanprakash
"""
from itertools import combinations
from copy import deepcopy
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from OrientationModels import *
from ConditionalIndependenceTest import *
   
def estimate_skeletal_graph(data):
    """
    Recovers a skeletal graph from observational data which will then be used to infer a DAG
    by orienting edges(and possibly, removing cycles).
    
    PARAMETERS
    ----------
    - data (pandas DataFrame) : the input observational data of shape (N, D)
    
    RETURNS
    -------
    - a skeletal (undirected) graph inferred from conditional independence tests
    - seperation set (dict) : the minimum seperation set for every node
    """
    # first, start with a complete undirected graph
    G = nx.Graph()
    G.add_nodes_from(data.columns)
    G.add_edges_from(list(combinations(data.columns, r=2)))

    def _check_termination(graph, n):
        """
        Checks if terminating condition for the skeleton discovery algorithm is satisfied.
        (Local Helper function)
        
        PARAMETERS
        ----------
        - graph (nx.Graph object) : the undirected graph
        - adj_set (dict) : the adjacency list for every node in <graph>
        - n (int) : the size of the conditioning set
        
        RETURNS
        -------
        - Boolean (if True, algorithm should terminate).
        """
        for (i, j) in graph.edges:
            if len(list(graph.neighbors(i)))-1>n:
                return False
        return True 
    
    # next, perform conditional independence tests to remove edges
    c_size = 0 # size of conditioning set
    sep_set = {} # seperation sets for every adjacent (i, j) nodes 
    while True:
        edge_pairs = list(G.edges)
        for (i, j) in edge_pairs:
            l = len(list(G.neighbors(i)))-1 # number of adjacent nodes for node 'i' excluding node 'j'
            if l>=c_size:
                if c_size==0:
                    conditioning_sets = [[]]
                else:
                    t = list(G.neighbors(i))
                    t.remove(j)
                    conditioning_sets = combinations(t, r=c_size)
                for Z in conditioning_sets:
                    CI = test_conditional_independence(x=data[i].values[:, np.newaxis], \
                                                       y=data[j].values[:, np.newaxis], \
                                                       z=data[list(Z)].values, threshold=0.01)
                    if CI:
                        # remove parallel edges and add conditioning set to the separation set
                        G.remove_edge(i, j)
                        sep_set[(i,j)] = sep_set[(j,i)] = Z
                        break
        if _check_termination(G, c_size):
            break
        c_size+=1
    return G, sep_set
    
def infer_DAG(data, orientation_model='Tree'):
    """
    Infers a DAG from observational data, using GNNs.
    
    PARAMETERS
    ----------
    - data (pandas DataFrame) : the input observational data of shape (N, D)
    
    RETURNS
    -------
    - a DAG ; nx.DiGraph object with no cycles
    """
    graph_skeleton, seperation_set = estimate_skeletal_graph(data)
    if orientation_model=='Tree':
        model = OrientationTree()
    elif orientation_model=='Net':
        model = OrientationNet(batch_size=10, lr=0.001, num_hidden_layers=1, \
                               nh_per_layer=[64], training_epochs=1000, \
                               test_epochs=100, max_iterations=1, runs_per_iteration=2)
    else:
        raise ValueError('Invalid argument value for `orientation_model`!')
    return orient_undirected_edges(model, data, graph_skeleton)

def orient_undirected_edges(model, data, skeleton):
    """
    Orients the edges of a skeletal graph.
    
    PARAMETERS
    ----------
    - data (pandas DataFrame) : the input observational data of shape (N, D)
    - skeleton (nx.Graph object) : the skeletal graph with no directed edges
    
    RETURNS
    -------
    - a Directed graph with possible cycles ; nx.DiGraph object.
    """
    g = nx.DiGraph()
    g.add_nodes_from(skeleton.nodes)
    edge_pairs = list(skeleton.edges)
    for (i, j) in edge_pairs:
        input_data = data[[i, j]].values
        try:
            model.reset()
        except:
            pass
        score = model._compute_direction_score(input_data)
        if score<0:
            g.add_edge(j, i, weight=abs(score))
        else:
            g.add_edge(i, j, weight=score)
    return remove_cycles(g) 

def remove_cycles(DiG):
    """
    Removes cycles in a directed graph based on the heuristic : 
        ' Edge with minimum score in every cycle is reversed and if possible, removed'.
    
    PARAMETERS
    ----------
    - DiG (nx.DiGraph object) : the directed graph
    
    RETURNS
    -------
    - a DAG.
    """
    ncycles = len(list(nx.simple_cycles(DiG)))
    while not nx.is_directed_acyclic_graph(DiG):
        cycle = next(nx.simple_cycles(DiG))
        # initialize collections of edges and corresponding scores contained in this cycle
        edges = [(cycle[-1], cycle[0])]
        scores = [(DiG[cycle[-1]][cycle[0]]['weight'])]
        for i, j in zip(cycle[:-1], cycle[1:]):
            edges.append((i, j))
            scores.append(DiG[i][j]['weight'])
        i, j = edges[scores.index(min(scores))]
        DiG_ = deepcopy(DiG)
        DiG_.remove_edge(i, j)
        DiG_.add_edge(j, i)
        temp = len(list(nx.simple_cycles(DiG_)))
        if temp<ncycles:
            DiG.add_edge(j, i, weight=min(scores))
        DiG.remove_edge(i, j)
        ncycles = temp
    return DiG
        
if __name__=='__main__':
    data = np.zeros((10000, 4))
    data[:, 0] = np.random.normal(loc=10.0, scale=5.0, size=10000)
    data[:, 1] = np.random.normal(loc=1.0, scale=2.0, size=10000)
    data[:, 2] = np.random.gamma(2, 0.65, 10000)
    data[:, 3] = data[:, 1]+data[:, 2]
    df = pd.DataFrame(data, columns=['A', 'B', 'C', 'D'])
    G = infer_DAG(df, orientation_model='Tree')
    nx.draw(G, with_labels=True)
    plt.show()