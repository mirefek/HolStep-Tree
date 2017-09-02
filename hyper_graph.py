#!/usr/bin/python3.5

import numpy as np

class HyperGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def nodes_num(self): return len(self.nodes)

    def new_node(self, data):
        self.nodes.append(data)
        return self.nodes_num()-1

    def add_edge(self, edge_type, edge_nodes):
        edge = len(self.edges)
        self.edges.append((edge_type, edge_nodes))

    def to_simple_graph(self):
        simple_nodes = [set() for node in self.nodes]
        for edge in self.edges:
            for n1 in edge[1]:
                for n2 in edge[1]:
                    if n1 != n2: simple_nodes[n1].add(n2)

        return [np.array(list(node)) for node in simple_nodes]

    def partition(self, num_clusters, partition):

        factor_graph = HyperGraph()
        factor_graph.nodes = [[] for node in range(num_clusters)]
        for node,data in enumerate(self.nodes):
            factor_graph.nodes[partition[node]].append(data)

        factor_edges = set(
            (edge_type, tuple(partition[node] for node in edge_nodes))
            for edge_type, edge_nodes in self.edges
        )
        def is_nonconstant_edge(edge_type_nodes):
            edge_nodes = edge_type_nodes[1]
            for node in edge_nodes:
                if node != edge_nodes[0]: return True
            return False

        factor_graph.edges = list(filter(is_nonconstant_edge, factor_edges))

        return factor_graph
