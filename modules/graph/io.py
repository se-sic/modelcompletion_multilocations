"""
This module includes convenience functions (API) for loading and storing graphs to disk.
"""
import json

import os
import pathlib
from typing import List
import networkx as nx
from networkx.readwrite import json_graph
import modules.textual.edgel_utils as edgel
import modules.textual.io
from modules.textual.io import import_tlv, serialize_edgeL_database

# TODO implement heuristic to automatically recognize graph format

EDGEL = 0
LG = 1
NX_JSON = 2

def read_database(path: any, is_directed=True, format: int = LG) -> List[nx.Graph]:
    graphs = None
    if (format == EDGEL):
        graphs = [graph for correct, graph in modules.textual.io.import_database(path, is_directed=is_directed, version=edgel.V_ORIGINAL) if correct]
    elif (format == LG):
        graphs = import_tlv(path, is_directed=is_directed, parse_support=False) 
    elif( format == NX_JSON):
        with open(path, 'r') as f:
            data = json.load(f)
            graphs = [json_graph.node_link_graph(d) for d in data]
    
    return graphs

def _graph_to_dict(graph, graph_id):
    data = {
        "id": graph_id,
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": [{"id": str(node), "label": str(attr['label']), "color": attr.get('color', 'grey')} for node, attr in graph.nodes(data=True)],
        "links": [{"source": str(edge[0]), "target": str(edge[1]), "label": str(edge[2]['label']), "color": edge[2].get('color', 'grey')} for edge in graph.edges(data=True)]
    }
    return data

def _graph_to_dict_general(graph, graph_id):
    data = {
        "id": graph_id,
        "directed": True,
        "multigraph": False,
        "graph": {},
        "nodes": [{"id": str(node), **{key: str(value) for key, value in attr.items()}} for node, attr in graph.nodes(data=True)],
        "links": [{"source": str(edge[0]), "target": str(edge[1]), **{key: str(value) for key, value in edge[2].items()}} for edge in graph.edges(data=True)]
    }
    return data

def save_graphs(graphs: List[nx.Graph], path_results: any):
    """Serialize as nx Json as well as edgeL database

    Args:
        graphs (List[nx.Graph]): _description_
        path_results (pathlib.Path): _description_
    """
    os.makedirs(path_results, exist_ok=True)
    graph_dicts = [_graph_to_dict(graph, i) for i, graph in enumerate(graphs)]

    json_file_path = os.path.join(path_results, "graphs.json")

    with open(json_file_path, 'w') as f:
        json.dump(graph_dicts, f, indent=4)


    serialize_edgeL_database(graphs, path_results,  single_file=True)
    
    
def save_components_networkx(graph: nx.Graph, path_results: any, filename: str):
    """Serialize as nx Json 

    Args:
        graphs (List[nx.Graph]): _description_
        path_results (pathlib.Path): _description_
    """
    os.makedirs(path_results, exist_ok=True)
   
    graph_dicts = _graph_to_dict_general(graph, graph.diff_id)

    json_file_path = os.path.join(path_results,filename)

    with open(json_file_path, 'w') as f:
        json.dump(graph_dicts, f, indent=4)

    
    
def load_components_networkx(data_folder, mark_filename = False):
    components = []
    for filename in os.listdir(data_folder):
        if not filename.endswith('.json'):
            continue
        with open(os.path.join(data_folder, filename), 'r') as f:  # open in readonly mode
            json_str = f.read()
            data = json.loads(json_str)
            f.close()
            H = json_graph.node_link_graph(data)
            if (mark_filename): 
                H.diff_id=filename
            components.append(H)
    return components



def evaluate_and_save_graphs_during_training(graph_data: List[nx.Graph], path_results: pathlib.Path,
                                             check_meta_model_validity=False, epoch=-1):
    if (epoch != -1):
        path_results = path_results + "epoch_" + str(epoch) + "/"

    # calculate_graph_metrics(graph_data, check_meta_model_validity)
    save_graphs(graph_data, path_results)

