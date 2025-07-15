from typing import List

import networkx as nx
import torch

INPUT_FORMAT_NX = "NX"
INPUT_FORMAT_LG = "LG"

def graph_representation_to_digraph(node_types, adjacency_triu, lookuptable, is_non_int, round_number=-1, graph_number=-1):


    di_graph = nx.DiGraph()
    di_graph.name = graph_number

    delete_nodes = []

    # Add nodes to nx.DiGraph
    for i in range(len(node_types)):
        if node_types[i].nelement() == 1 and node_types[i].item() == 0:
            #label = "None"
            delete_nodes.append(i)
            continue
        else:
            if (is_non_int):
                label = node_types[i]

            else:
                label = lookuptable.node_labels[int(node_types[i])]

            di_graph.add_node(i, label=label)

    # Add edges to nx.DiGraph
    adjacency_matrix = torch.reshape(adjacency_triu, (len(node_types),len(node_types)))

    for ix, row in enumerate(adjacency_matrix):
        if ix in delete_nodes:
            # There can not be any edges to the "None" nodes
            continue
        for iy, label_type in enumerate(row):
            if iy in delete_nodes:
                # There can not be any edges to the "None" nodes
                continue
            if int(label_type) == 0:
                # We exclude the "None" edges
                continue
            di_graph.add_edge(ix, iy, label=lookuptable.edge_labels[int(label_type)])

    return di_graph


def connected_components(graph: nx.Graph) -> List[nx.Graph]:
    if nx.is_directed(graph):
        components = [graph.subgraph(c).copy() for c in nx.weakly_connected_components(graph)]
    else:
        components = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]

    connected_component_id =0
    for c in components:
        c.diff_id = graph.diff_id
        c.folder_name= graph.folder_name
        c.component_id = connected_component_id
        connected_component_id+=1


    return components


def count_edges(graph: nx.Graph) -> int:
    return len(graph.edges())


def count_edges_db(graph_db: List[nx.Graph]) -> int:
    return sum([count_edges(graph) for graph in graph_db])


class FilterConfig():
    def __init__(self, filter_too_large_nb_nodes=25, filter_too_large_nb_edges=40, filter_too_many_similar_max_similar=2, filter_too_many_similar_max_nodes=10):
        self.filter_too_large_nb_nodes = filter_too_large_nb_nodes
        self.filter_too_large_nb_edges = filter_too_large_nb_edges
        self.filter_too_many_similar_max_similar = filter_too_many_similar_max_similar
        self.filter_too_many_similar_max_nodes = filter_too_many_similar_max_nodes

# Filters components with more than nb_nodes/nb_edges nodes/edges. Use -1 for infinity.
def filter_too_large(components: list, id_diffgraphs_per_compo_list, filtered: dict, nb_nodes, nb_edges):
    new_components = []
    new_ids =[]
    component_id=0
    for component in components:
        if not (nb_nodes != -1 and (component.number_of_nodes() > nb_nodes or component.number_of_edges() > nb_edges)):
            new_components.append(component)
            new_ids.append(id_diffgraphs_per_compo_list[component_id])
        component_id +=1


    filtered["too_large"] = len(components)-len(new_components)
    #print("Filtered out %d components that are too large, i.e., more than %d nodes or %d edges" % (filtered["too_large"], nb_nodes, nb_edges))
    return new_components, new_ids, filtered


def filter_too_many_similar_nodes(components: list, filtered: dict, max_similar=2, max_nodes=10):
    new_components = []

    for component in components:
        labels = label_count_for_component(component)
        # if there are more than max_similar node labels with more than max_nodes
        if not (np.sum(np.array(list(labels.values())) > max_nodes) > max_similar):
            new_components.append(component)

    filtered["too_many_similar"] = len(components)-len(new_components)
    #print("Filtered out %d components with too many similar nodes, i.e., more than %d labels appeared more than %d times" % (filtered["too_many_similar"] , max_similar, max_nodes))
    return new_components, filtered


def label_count_for_component(component):
    labels = {}
    for node in component.nodes(data=True):
        if node[1]['label'] in labels.keys():
            labels[node[1]['label']] += 1
        else:
            labels[node[1]['label']] = 1
    return labels


def has_node(graph, label):

    return label in [node[1]['label'] for node in list(graph.nodes(data=True))]


def split_in_connected_components(graphs):
    components = []
    id_diffgraphs_per_compo =[]
    nb_of_components_per_diff = []
    for graph in graphs:
        new_components = connected_components(graph)

        nb_of_components_per_diff.append(len(new_components))
        components += new_components
        id_diffgraphs_per_compo += [graph.diff_id] * len(new_components)

    return components, nb_of_components_per_diff, id_diffgraphs_per_compo


