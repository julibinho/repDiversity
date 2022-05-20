#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import random
import Levenshtein as lv
import matplotlib.pyplot as plt
import csv
import pandas as pd
import networkx as nx
from scipy.spatial import distance
from sklearn.metrics import jaccard_score
from polyleven import levenshtein
from pyvis.network import Network
from scipy.spatial import distance
import json
from networkx.readwrite import json_graph


""" Repertoire Summary """
def hill_diversity(x, alpha = np.arange(0, 10, 0.1), hill = False, clone_id=True) :
    """
    Calculate Hill Diversity Index for alpha in [0,1] with a step of 0.1 by default
    input1 x : repertoire as dataframe
    input2 alpha : list of alpha value to compute hill diversity indice
    input3 hill : boolean to pass the computed values to exponential
    input4 clone_id : if True, take the clone_id column as input
    output1 : list of hill diversity indices
    """
    if clone_id==True :
        f = np.array(x.clone_id.value_counts())
    else : 
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

### TEST : Other distance but too slow ###
def build_kmers(sequence, ksize):
    kmers = []
    n_kmers = len(sequence) - ksize + 1

    for i in range(n_kmers):
        kmer = sequence[i:i + ksize]
        kmers.append(kmer)

    return kmers

def kmer_dist(seq1, seq2, ksize) :
    seq1_kmers = set(build_kmers(seq1, ksize))
    seq2_kmers = set(build_kmers(seq2, ksize))
    all_kmers = seq1_kmers | seq2_kmers
    if not all_kmers :
        return np.nan
    shared_kmers = seq1_kmers & seq2_kmers
    nb_unique = len(all_kmers) - len(shared_kmers)
    fraction_unique = nb_unique / len(all_kmers)
    
    return fraction_unique
#######


def PDD(x, column = 'sequence_alignment', approx = True, n = 1000) :
    """
    Pairwise Distance Distribution
    input1 x : repertoire as panda dataframe
    input2 column : column on which the sequence are, sequence_alignment as default
    input3 approx : if True, sample the repertoire 10 times 
    imput4 n : size of sample
    output1 vect : vector that contains levenshtein distance for each pair of words, can be plot with plt.hist
    """
    if approx == False :
        X = x[column]
    elif approx == True :
        X = x[column]
        if len(X) > 10000 :
            X = x[column].sample(10000)
    L_ref = len(X)
    vect = []
    while len(X) > n :
        sample = random.sample(list(X.index), n)
        print('progression : ', L_ref - len(X), '/', L_ref)
        sample_X = X[sample]
        index = list(sample_X.index)
        L = len(index)
        while index != [] :
            for i in sample_X.index :
                #print('progession : ', L - len(index)+1,' / ' , L)
                index.remove(i)
                for j in index :
                    #vect.append(levenshtein(X[i], X[j]))
                    vect.append(lv.distance(sample_X[i], sample_X[j]))
                    
        X = X.drop(sample)
        
    return vect


# count values to plot
def count_values(x) :
    """
    convert list of values into histogram vector, 
    used in JSD function
    input x : list of values
    output vect : histogram vector
    """
    vect = []
    for i in range(np.max(x)) :
        vect.append(np.count_nonzero(np.array(x) == i))
        
    return vect


def network(x, threshold = 20, option = 'sequence', approx = True, n = 1000, show = True) :
    """
    Establish a network which nodes represents sequences and edges link sequences 
    with Levenshtein distance <= threshold
    input1 x : repertoire file as panda dataframe
    imput2 threshold : maximum levenshtein distance to link nodes
    input3 option : column to consider, sequence as default
    input4 approx : sample the repertoire if True
    input5 n : size of sample
    input6 show : if True, show the graph in a html window
    output : graph informations, average degree, number of connected components, size of connected components
    """
    
    if approx == False :
        X = x
    elif approx == True :
        if len(x) > 10000 :
            X = x.sample(n)
        else :
            X = x
    
    # get frequencies of each sequences
    y = X[option].value_counts()
    
    f = y.index
    f = np.array(f)
    G = nx.Graph()
    
    for i in range(len(f)) :
        print('progression : ', i+1 ,'/', len(f)) 
        if f[i] not in G.nodes :
            G.add_node(f[i], label = str(i), size = int(y[f[i]]))
        for j in range(i+1, len(f)) :
            if f[j] not in G.nodes :
                G.add_node(f[j], label = str(j), size = int(y[f[j]]))
            dist = lv.distance(f[i], f[j])
            if dist <= threshold :
                G.add_edge(f[i], f[j], length = int(dist)) 
    
    if show == True :
        nx.draw(G)
        net = Network('1000px', '1000px')
        net.from_nx(G)
        net.show_buttons(filter_=["physics"])
        net.show('nx.html')
    
    avg_deg = 0
    for i in G.nodes :
        avg_deg += G.degree(i)
    avg_deg = avg_deg / len(G.nodes)
    
    list_size = [len(c) for c in sorted(nx.connected_components(G), key=len, reverse=True)]
    
    return nx.info(G), avg_deg, nx.number_connected_components(G), list_size


def network_json(x, threshold = 10, option='sequence', approx=True, n=5000) :
    """
    return a json formatted graph to visualize with d3.js
    input1 x : repertoire file as panda dataframe
    imput2 threshold : maximum levenshtein distance to link nodes
    input3 option : column to consider, sequence as default
    input4 approx : sample the repertoire
    input5 n : size of sample
    output : dict representing the graph with json format
    """
    if approx == False :
        X = x
    elif approx == True :
        if len(x) > 10000 :
            X = x.sample(n)
    
    # get frequencies of each sequences
    y = X[option].value_counts()
    
    f = y.index
    f = np.array(f)
    G = nx.Graph()
    
    for i in range(len(f)) :
        print('progression : ', i+1 ,'/', len(f)) 
        if f[i] not in G.nodes :
            G.add_node(f[i], label = str(i), size = int(y[f[i]]))
        for j in range(i+1, len(f)) :
            if f[j] not in G.nodes :
                G.add_node(f[j], label = str(j), size = int(y[f[j]]))
            dist = lv.distance(f[i], f[j])
            if dist <= threshold :
                G.add_edge(f[i], f[j], length = int(dist)) 
    
    dict_graph = json_graph.node_link_data(G)
    
    return dict_graph


def vdj_usage(x) :
    """
    return V,D,J genes usage
    input1 x : repertoire as dataframe
    output : 3 dictionnaries that contains V,D,J genes names 
    """
    # count each V,D,J genes
    v_serie = x.v_call.value_counts()
    d_serie = x.d_call.value_counts()
    j_serie = x.j_call.value_counts()
    
    # convert each series as dict
    v_dict = {}
    d_dict = {}
    j_dict = {}
    for i,j in zip(v_serie.index, v_serie.values) :
        v_dict[i] = j
    for i,j in zip(d_serie.index, d_serie.values) :
        d_dict[i] = j
    for i,j in zip(j_serie.index, j_serie.values) :
        j_dict[i] = j
    
    return v_dict, d_dict, j_dict

def plot_vdj_hist(x, threshold=100) :
    """
    plot histogram of V and J gene usage
    input1 x : output of vdj_usage()
    input2 threshold : threshold value, genes which have a value under this threshold will not be considered
    output : None
    """
    v_usage_week1 = {}
    for key in x[0].keys() :
        if x[0][key] > threshold :
            v_usage_week1[key] = x[0][key]

    plt.figure(112, figsize=(15,8))
    plt.bar(v_usage_week1.keys(), v_usage_week1.values())
    plt.xticks(rotation=90, fontsize=15);
    plt.title('IGHV Gene Usage', fontsize=20);
    plt.xlabel('Gene ID', fontsize=20)
    plt.ylabel('Occurence', fontsize=20)

    # Sort J Gene
    j_usage_week1 = {}
    for key in x[2].keys() :
        if x[2][key] > threshold :
            j_usage_week1[key] = x[2][key]

    plt.figure(113, figsize=(15,8))
    plt.bar(j_usage_week1.keys(), j_usage_week1.values(), color='orange')
    plt.xticks(rotation=90, fontsize=15);
    plt.title('IGHJ Gene Usage', fontsize=20);
    plt.xlabel('Gene ID', fontsize=20)
    plt.ylabel('Occurence', fontsize=20)
    
    
    return



""" Repertoire Comparison """
def cor(list_rep) :
    """
    Calculate the Pearson correlation coefficient between several vectors of diversity
    obtained with hill_diversity() function
    input : list of repertoire as list of dataframe
    output : Pearson coefficient correlation matrix
    """
    
    list_div = []
    
    for i in list_rep :
        list_div.append(hill_diversity(i))
    

    r = np.corrcoef(list_div)
    
    return r


def JSD(x, y) :
    """
    Calculate Jensen-Shannon Divergence between two distance pairwise distribution 
    input1 x : first repertoire 
    input2 y : second repertoire
    output : Jensen-Shannon Divergence
    """
    
    pdd_x = PDD(x)
    pdd_y = PDD(y)
    
    x_vect = count_values(np.array(pdd_x))
    y_vect = count_values(np.array(pdd_y))
    
    # if vectors are not of the same size, we complete the littlest with zeros
    if len(x_vect) > len(y_vect) :
        for i in range(len(x_vect)-len(y_vect)) :
            y_vect.append(0)
    elif len(y_vect) > len(x_vect) :
        for i in range(len(y_vect)-len(x_vect)) :
            x_vect.append(0)
    
    JSD = distance.jensenshannon(x_vect, y_vect)
    
    return JSD**2


def JSD_matrix(list_rep) :
    """
    Calculate the JSD value for each pair of repertoire
    input list_rep :list of repertoire dataframe
    output : matrix containing JSD values 
    """
    
    N = len(list_rep)
    
    jsd_mat = np.zeros((N, N))
    
    for i in range(N) :
        #print(i)
        for j in range(i+1, N) :
            jsd_mat[i][j] = JSD(list_rep[i], list_rep[j])
            
    return jsd_mat



def overlap(x, y, option='sequence') :
    """
    Calculate the repertoire overlap value between two repertoires
    option argument defines which column is used, 'sequence' by default
    input1 x : first repertoire
    input2 y : second repertoire
    input3 option : name of which column is used
    output value : repertoire overlapv alue
    """
    # delete duplicate
    x_rep = np.array(x.drop_duplicates(subset=[option])[option])
    y_rep = np.array(y.drop_duplicates(subset=[option])[option])
    
    C = x_rep[np.in1d(x_rep, y_rep)]
    
    value = len(C) / min(len(x_rep), len(y_rep))
    
    return value



def overlap_matrix(list_rep, option = 'sequence') :
    """
    Calculate the repertoire overlap value for eawh pair of repertoire in list_rep
    input1 list_rep : list of repertoire dataframe
    input2 option : name of which colmun is used, 'sequence' by default
    output overlap_mat : matrix containing overlap values 
    """
    N = len(list_rep)
    
    overlap_mat = np.zeros((N,N))
    
    for i in range(N) :
        for j in range(N) :
            overlap_mat[i][j] = overlap(list_rep[i], list_rep[j])
            
    return overlap_mat


# VDJ Comparison
def vdj_comparison(x, y) :
    """
    return dataframe of V and J genes in common
    input1 x : first repertoire as dataframe
    input2 y : second repertoire as dataframe
    output : dataframe of in common V and J genes 
    """
    
    # vdj_usage
    vdj_x = vdj_usage(x)
    vdj_y = vdj_usage(y)
    
    # select in common genes
    v_incom = vdj_x[0].keys() & vdj_y[0].keys()
    j_incom = vdj_x[2].keys() & vdj_y[2].keys()
    
    # Construct dataframe
    v_dataframe = {'rep1': [], 'rep2': []}
    j_dataframe = {'rep1': [], 'rep2': []}
    v_dataframe['rep1'] = [vdj_x[0][key] for key in v_incom]
    v_dataframe['rep2'] = [vdj_y[0][key] for key in v_incom]
    j_dataframe['rep1'] = [vdj_x[2][key] for key in j_incom]
    j_dataframe['rep2'] = [vdj_y[2][key] for key in j_incom]
    index_v = list(v_incom)
    index_j = list(j_incom)
    v_dataframe = pd.DataFrame(v_dataframe, index_v)
    j_dataframe = pd.DataFrame(j_dataframe, index_j)
    
    return v_dataframe, j_dataframe


def plot_vdj_comparison(dataframe, threshold=100) :
    """
    Plot histogram of V and J genes
    input1 dataframe : output of vdj_comparison()
    input2 threshold : trehsold value to print the gene
    output : None
    """
    
    new_v3 = []
    row_v3 = []
    for i in dataframe[0].iterrows() :
        for value in i[1].values :
            if value > threshold :
                new_v3.append(i[1].values)
                row_v3.append(i[0])
                break

    new_v3 = np.array(new_v3)


    new_j3 = []
    row_j3 = []
    for i in dataframe[1].iterrows() :
        for value in i[1].values :
            if value > threshold :
                new_j3.append(i[1].values)
                row_j3.append(i[0])
                break

    new_j3 = np.array(new_j3)
    
    
    new_v3 = pd.DataFrame(new_v3, columns=['rep1', 'rep2'], index=row_v3)
    new_j3 = pd.DataFrame(new_j3, columns=['rep1', 'rep2'], index=row_j3)
    
    list_dataframe = [new_v3, new_j3]
    
    
    # Plot 
    list_dataframe[0].plot.bar(subplots=True, figsize=(15,8))
    plt.xlabel('IGHV Gene ID', fontsize=20)
    plt.ylabel('Occurence', fontsize=20)
    plt.xticks(fontsize = 20)
    
    list_dataframe[1].plot.bar(subplots=True, figsize=(15,8))
    plt.xlabel('IGHJ Gene ID', fontsize=20)
    plt.ylabel('Occurence', fontsize=20)
    plt.xticks(fontsize = 20)
    
    return
    

def vdj_comparison_3(x, y, z) :
    """
    return dataframe of V and J genes in common
    input1 x : first repertoire as dataframe
    input2 y : second repertoire as dataframe
    input3 z : third repertoire as dataframe
    output : dataframe of in common V and J genes 
    """
    
    # vdj_usage
    vdj_x = vdj_usage(x)
    vdj_y = vdj_usage(y)
    vdj_z = vdj_usage(z)
    
    # select in common genes
    v_incom = vdj_x[0].keys() & vdj_y[0].keys() & vdj_z[0].keys()
    j_incom = vdj_x[2].keys() & vdj_y[2].keys() & vdj_z[2].keys()
    
    # Construct dataframe
    v_dataframe = {'week0': [], 'week1': [], 'week2' : []}
    j_dataframe = {'week0': [], 'week1': [], 'week2' : []}
    v_dataframe['week0'] = [vdj_x[0][key] for key in v_incom]
    v_dataframe['week1'] = [vdj_y[0][key] for key in v_incom]
    v_dataframe['week2'] = [vdj_z[0][key] for key in v_incom]
    j_dataframe['week0'] = [vdj_x[2][key] for key in j_incom]
    j_dataframe['week1'] = [vdj_y[2][key] for key in j_incom]
    j_dataframe['week2'] = [vdj_z[2][key] for key in j_incom]
    index_v = list(v_incom)
    index_j = list(j_incom)
    v_dataframe = pd.DataFrame(v_dataframe, index_v)
    j_dataframe = pd.DataFrame(j_dataframe, index_j)
    
    return v_dataframe, j_dataframe


def plot_vdj_comparison_3(dataframe, threshold=100) :
    """
    Plot histogram of V and J genes to compare 2 repertoires
    input1 dataframe : output of vdj_comparison_3()
    input2 threshold : threshold value to plot the gene
    output : None
    """
    
    new_v3 = []
    row_v3 = []
    for i in dataframe[0].iterrows() :
        for value in i[1].values :
            if value > threshold :
                new_v3.append(i[1].values)
                row_v3.append(i[0])
                break

    new_v3 = np.array(new_v3)


    new_j3 = []
    row_j3 = []
    for i in dataframe[1].iterrows() :
        for value in i[1].values :
            if value > threshold :
                new_j3.append(i[1].values)
                row_j3.append(i[0])
                break

    new_j3 = np.array(new_j3)
    
    
    new_v3 = pd.DataFrame(new_v3, columns=['week0', 'week1', 'week2'], index=row_v3)
    new_j3 = pd.DataFrame(new_j3, columns=['week0', 'week1', 'week2'], index=row_j3)
    
    list_dataframe = [new_v3, new_j3]
    
    
    # Plot 
    list_dataframe[0].plot.bar(subplots=True, figsize=(15,8))
    plt.xlabel('IGHV Gene ID', fontsize=20)
    plt.ylabel('Occurence', fontsize=20)
    plt.xticks(fontsize = 20)
    
    list_dataframe[1].plot.bar(subplots=True, figsize=(15,8))
    plt.xlabel('IGHJ Gene ID', fontsize=20)
    plt.ylabel('Occurence', fontsize=20)
    plt.xticks(fontsize = 20)
    
    return
