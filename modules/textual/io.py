import json
import logging
import os
import pathlib
import re
from typing import List, Tuple, Callable

import networkx as nx
import pandas as pd

from modules.textual.edgel_utils import V_ORIGINAL, _split_string, parse_graphs, dfs_edges, serialize_graph

LOGGER = logging.getLogger()
def transform_labels(label: str, element_type: str = 'object') -> str:
    """Transforms a label of the original attribute less format to the JSON-like format.

    Args:
        token (str): label serialization in the original format.

    Returns:
        str: label serialization in the JSON-like format.
    """
    change_type, obj_type = label.split('_')

    if element_type == 'object':
        element_name_identifier = 'className'
    elif element_type == 'reference':
        element_name_identifier = 'referenceTypeName'
    else:
        LOGGER.error(f"ivalid element type name while migrating EdgeL format: {element_type}")
        raise Exception(f"ivalid element type name while migrating EdgeL format: {element_type}")

    return json.dumps({
        'changeType': change_type,
        'type': element_type,
        element_name_identifier: obj_type,
        'attributes': {}
    })
def auto_migrate_label(auto_migrate: bool, label: str, element_type: str = 'object') -> dict:
    if not auto_migrate:
        return label

    label = label.strip("'")
    regex_old_label = r"(Add|Remove|Preserve)_(.*)"
    match_old_format = re.match(regex_old_label, label)
    if match_old_format:
        label = transform_labels(label=label, element_type=element_type)
    return label
########################### Line Graph Parsers/Serializers #####################################################
# Export TLV (e.g., gSpan consumes TLV)
def export_TLV(graph_db,id_diffgraphs_per_compo, path):
    f = open(path, 'w')
    for idx, graph in enumerate(graph_db):
        if graph.name is None or graph.name == "":
            graph.name = str(idx)

        id_diffgraphs= str(id_diffgraphs_per_compo[idx])
        assert(id_diffgraphs==str(graph.diff_id))

        f.write(f"t # {graph.name} : {id_diffgraphs} \n")
        temp_graph = nx.convert_node_labels_to_integers(graph, first_label=0)
        # sort indices
        vertices = temp_graph.nodes(data=True)
        for node, data in vertices:
            if 'label' not in data.keys():
                print("WARN: Unlabeled nodes in graph data for graph %s." % graph.name)
                label = "UNKNOWN_LABEL"
            else:
                label = data['label']

            f.write("v " + str(node) + " " + repr(label) + '\n')
        edges = temp_graph.edges(data=True)
        for source, target, data in edges:
            if 'label' not in data.keys():
                print("WARN: Unlabeled edges in graph data for graph %s." % graph.name)
                label = "UNKNOWN_LABEL"
            else:
                label = data['label']
            f.write("e " + str(source) + " " + str(target) + " " + repr(label) + '\n')
    f.close()


def import_tlv_folder(folder_path, postfix='.lg', is_directed=True, parse_support=True):
    '''
    See import_tlv. Just iterates over the folder and concats.
    Doesn't take graph isomorphisms into account, i.e., isomorphic graphs in different files could appear as duplicates.
    '''
    if parse_support:
        graphs = {}
    else:
        graphs = []

    for file in os.listdir(folder_path):
        if file.endswith(postfix):
            new_graphs = import_tlv(os.path.join(folder_path, file), is_directed, parse_support)
            if parse_support:
                graphs.update(new_graphs)
            else:
                graphs += new_graphs
    return graphs


def import_tlv(path, is_directed=True, parse_support=True, auto_migrate=True):
    '''
    Parses the given file as line graph and (optionally) parses the support of the graph. There are multiple different formats to obtain the support from.
    It returns a dictionary of graphs with their support or a list of all graphs if no support parsing is desired.

    params: path, the path to the line graph file
            parse_support: true, if support should also be parsed
            auto_migrate: whether the old file format should be upgraded to the new one. Defaults to True.
    returns:  a dictionary dict(DiGraph, int) with the graphs and their suppor, if parse_support=True, else a list of graphs
    '''
    graph_db = open(path, 'r')
    next_line = graph_db.readline()
    if parse_support:
        graphs = {}
    else:
        graphs = []
    regex_header = r"t # (.*) : (.*) "
    regex_node = r"v (\d+) (.+).*"
    regex_edge = r"e (\d+) (\d+) (.+).*"

    # Some file formats give the support directly, others list all the embeddings. We support both options.
    regex_support = r"Support: (\d+).*"
    regex_embedding = r"#=> ([^\s]+) .*"

    # if tlv header continue parsing
    match_header = re.match(regex_header, next_line)
    if match_header:
        next_line = graph_db.readline()
    else:
        print("Error parsing graph db. Expecting TLV.")
        return {}

    while next_line:
        if is_directed:
            graph = nx.DiGraph()
        else:
            graph = nx.Graph()
        graph.name = match_header.group(1)
        graph.diff_id = match_header.group(2)
        support_set = set()
        support = None
        match_header = None

        while next_line and not match_header:
            match_node = re.match(regex_node, next_line)
            match_edge = re.match(regex_edge, next_line)
            match_support = re.match(regex_support, next_line)
            match_embedding = re.match(regex_embedding, next_line)
            if match_node:
                label = auto_migrate_label(auto_migrate, str(match_node.group(2)), element_type='object')
                graph.add_node(int(match_node.group(1)), label=label)
            elif match_edge:
                label = auto_migrate_label(auto_migrate, str(match_edge.group(3)), element_type='reference')
                graph.add_edge(int(match_edge.group(1)), int(match_edge.group(2)), label=label)
            elif match_support:
                support = int(match_support.group(1))
            elif match_embedding:
                support_set.add(str(match_embedding.group(1)))
            next_line = graph_db.readline()
            if next_line:
                match_header = re.match(regex_header, next_line)

        if support_set is not None:
            graph.graph['embeddings'] = str(support_set)

        if (support is None and support_set == set() and parse_support):
            print("WARN: Error parsing line graph with graph support. Check format.")
        elif not parse_support:
            graphs.append(graph)
        elif support is not None:
            graphs[graph] = support
        else:
            support = len(support_set)
            graphs[graph] = support
        next_line = graph_db.readline()
    return graphs


########################### End: Line Graph Parsers #####################################################

def import_database(path: pathlib.Path,
                    is_directed:bool = True,
                    version: int = V_ORIGINAL, synthetic_dataset=False,
                    postfix:str = '.edgl') -> List[Tuple[bool, nx.Graph]]:
  """Parse a database of edgl graphs.

  Automatically takes into account whether the path is a file or a folder.
  See import_tlv. Just iterates over the folder and concats.
  Doesn't take graph isomorphisms into account, i.e.,
  isomorphic graphs in different files could appear as duplicates.

  Args:
      path (pathlib.Path): The path to the database file or folder.
      is_directed (bool, optional): if graphs should be considered directed or undirected. Defaults to True.
      version (str, optional): The label version. Defaults to V_ORIGINAL.
      postfix (str, optional): The postfix for files if a folder is parsed. Defaults to '.edgl'.


  Returns:
      List[Tuple[bool, nx.Graph]]: A list of tuples.
        First element of the tuple is true, if the graph could be parsed and Second element is the nx.Graph.
  """

  # Determine if path is file or folder
  is_file = path.is_file()

  if is_file:
    with open(path) as f:
      graph_db = f.read()

    # split
    graph_strings = _split_string(graph_db)


  else:
    graph_strings = []
    # Iterate over folder
    for file in os.listdir(path):
      if file.endswith(postfix):
        with open(path / file) as f:
          graph_strings.append(f.read())


  # trim whitespaces
  graph_strings = [serialized_graph.strip() for serialized_graph in graph_strings]
  return parse_graphs(graph_strings, is_directed=is_directed, version=version, synthetic_dataset=synthetic_dataset)



def serialize_edgeL_database(nx_database: List[nx.Graph], save_path: str, serialization_strategy: Callable[[nx.Graph],Tuple[nx.Graph, List[Tuple]]] = dfs_edges, single_file=False, is_completion =False, uncompleted_graph_set=dict()):
  ''' For a given database of networkX graphs, serializes them in the edgeL format and saves them.
  If single_file is True, all graphs are written to one file separated by an empty line. If single_file is False, there is one file per graph.
  '''
  os.makedirs(save_path, exist_ok=True)

  if single_file:
    graph_db_string = ''
  for graph in nx_database:
    if is_completion:
      serialized_graph,_ = serialize_graph(graph, serialization_strategy=serialization_strategy,is_completion=is_completion, serialized_nodes=uncompleted_graph_set[graph.name] )
    else:
      serialized_graph,graph_set = serialize_graph(graph, serialization_strategy=serialization_strategy,is_completion=is_completion, serialized_nodes=set())
      uncompleted_graph_set[graph.name]= graph_set

    if single_file:
      graph_db_string += ('$$\n' if is_completion else '') + serialized_graph +  '\n\n'
    else:
      with open(f'{save_path}/{graph.name}.edgel', 'a') as f:
        f.write( ('$$\n' if is_completion else '') + serialized_graph)
  if single_file:
    with open(f'{save_path}/database.edgel', 'a') as f:
        f.write(graph_db_string)

  return uncompleted_graph_set
