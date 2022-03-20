#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import Levenshtein as lv
import networkx as nx

def hill_diversity(x, alpha = np.arange(0, 10, 0.1), hill = False) :
    """
    Calculate Hill Diversity Index for alpha in [0,1] with a step of 0.1 by default
    input1 x : repertoire as dataframe
    input2 alpha : list of alpha value to compute hill diversity indice
    input3 hill : boolean to pass the computed values to exponential
    output1 : list of hill diversity indices
    """
    
    f = np.array(x.sequence.value_counts())
    values = []
    
    # convert x as a frequencies list if x is a count list
    if np.sum(f) != 1 :
        f = f / np.sum(f)
    
    for q in alpha :
        # we distinguish particular cases :
        if q == 0 :
            values.append(np.log(np.sum(f>0)))
        elif q == 1 :
            values.append( -np.sum(f*np.log(f)) )
        elif q == 2 :
            values.append( -np.log(np.sum(f**2)) )
        else :
            values.append( (np.log(np.sum(f**q))) / (1-q) )
            
    if hill == True :
        values = np.exp(values)
    
    return values

def PDDistribution(x, column = 'sequence_alignment') :
    """
    input1 x : repertoire as panda dataframe
    input2 column : column on which the sequence are, sequence_alignment as default
    output1 vect : vector that contains levenshtein distance for each pair of words
    """
    
    X = x[column]
    vect = []
    
    for i in range(len(X)) :
        for j in range(i+1, len(X)) :
            #vect.append(levenshtein(X[i], X[j]))
            vect.append(lv.distance(X[i], X[j]))
            
    return vect

def count_values(x):
    vect = []
    for i in range(np.max(x)) :
        vect.append(np.count_nonzero(x == i))
    return vect


def network(x, col, clone = False, option = 'sequence') :
    """
    Establish a network which nodes represents sequences and edges link sequences 
    with Levenshtein distance = 1
    input1 x : repertoire file as panda dataframe
    input2 col : color of graph, str
    input3 clone : if True we consider all the sequences
                   if False we delete duplicated sequences
    input4 option : column to consider, sequence as default
    output : average degree 
    """
    
    if clone == False :
        #delete duplicate for less complexity
        f = x.drop_duplicates(subset=[option])[option]
    else :
        f = x[option]
    f = np.array(f)
    G = nx.Graph()
    
    for i in range(len(f)) :
        for j in range(i+1, len(f)) :
            G.add_node(i)
            G.add_node(j)
            if lv.distance(f[i], f[j]) <= 1 :
                G.add_edge(i, j, weight = 1)
            
    
    nx.draw(G, with_labels= False, node_color=col)
    plt.show()
    
    # Average Degree
    avg_deg = 0
    for i in range(len(nx.degree(G))) :
        avg_deg += nx.degree(G)[i]
    avg_deg = avg_deg / len(nx.degree(G))
    print('Number of connected components : ', nx.number_connected_components(G))
    print('Average degree : ', avg_deg)
    
    return avg_deg

