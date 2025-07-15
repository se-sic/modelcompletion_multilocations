#!/usr/bin/python  

# TODO this class should be refactored (more object oriented). I.e., we have A graph and edge classes and each has a parsing and serialization capabilities.

import random
from typing import Callable, List, Set, Tuple, Union
# graph library
import networkx as nx

# regular expressions (for parsing)

# logging
import logging
import re

from modules.domain_specific.change_graph import ChangeGraphEdge, ChangeGraphNode
LOGGER = logging.getLogger("EdgeL-Utils")

# TODO ctinnes V_JSON Is actually only something available for change graphs. for arbitrary graphs we do not have this logic. This class here should only handle arbitrary graphs.

# FORMAT VERSION
V_ORIGINAL = 1 # Supports only string without whitespace characters for edge and node labels
V_JSON = 2 # Supports JSON labels for edges and nodes. Furthermore, "_" labels can be used to not repeat any label
AUTO_VERSION = 3 # Checks automatically first if it matches JSON then fallbacks to V_ORIGINAL

DUMMY_NODE_LABELS = ["_", "\"{}\"", "{}"] # Node labels used to avoid repetition in edge serializations

####################### Serialization strategy ###################

# dfs-serialization
def dfs_edges(graph: nx.Graph):
    # Get all nodes with in-degree zero (we need to start the dfs from there)
    # roots = [node for (node, val) in graph.in_degree() if val == 0]
    return graph, [(x, y, graph.get_edge_data(x, y)) for (x, y, dir) in
                   nx.algorithms.traversal.edgedfs.edge_dfs(graph, orientation='ignore')]  # , source=roots))

# Random Serialization strategy
def random_edges(graph: nx.Graph):
    nnodes = graph.number_of_nodes()
    mapping = dict(zip(range(nnodes), random.sample(range(nnodes), nnodes)))
    edges = list(graph.edges(data=True))
    random.shuffle(edges)
    return graph, edges

# as-is
def as_is(graph: nx.Graph):
    return graph, list(graph.edges(data=True))

####################### edgeL (edge list) serialization ###################




# TODO implement other strategies
def serialize_graph(graph: nx.Graph, serialization_strategy: Callable[[nx.Graph],Tuple[nx.Graph, List[Tuple]]] = dfs_edges, is_completion: bool=False, serialized_nodes=None, version=V_ORIGINAL) -> Tuple[str, Set[int]]:
  if serialized_nodes is None: # This is to avoid confusion if serialized_nodes, by default, is set to set(), then calling this method multiple times will keep the state, which leads to unexpected behaviour.
    serialized_nodes = set()
  graph_string = ''
  # Add header
  if not is_completion:
    graph_string += f't # {graph.name} {graph.diff_id}\n'
  # Serialize edges
  graph, edges = serialization_strategy(graph)
  for edge in edges:
    graph_string+=serialize_edge(edge, graph, serialized_nodes, version=version)+'\n'
  
  return graph_string, serialized_nodes


def serialize_edge(edge: Tuple, graph: nx.Graph, serialized_nodes=set(), version=V_ORIGINAL):
  if 'label' not in edge[2].keys():
    LOGGER.warning("Unlabeled edges in graph data for graph %s." % graph.name)
    label = "UNKNOWN_LABEL"
  else:
    label = edge[2]['label']

  graph_nodes_temp = graph.nodes(data=True)
  graph_nodes = {int(graph_node[0]): graph_node[1] for graph_node in graph_nodes_temp}
   
  if version == V_ORIGINAL:
    dummy_label = "_" 
  elif version == V_JSON:
    dummy_label = "\"{}\""
  else:
    LOGGER.warning(f"Unknown EdgL version: {version}")
    dummy_label = "_"

  if int(edge[0]) in serialized_nodes:
    src_label = dummy_label
  elif 'label' in graph_nodes[int(edge[0])].keys():
    src_label = graph_nodes[int(edge[0])]['label']
    serialized_nodes.add(int(edge[0]))
  else:
    LOGGER.warning("Unlabeled nodes in graph data for graph %s." % graph.name)
    src_label = "UNKNOWN_LABEL"
    serialized_nodes.add(int(edge[0]))
    
  if int(edge[1]) in serialized_nodes:
    tgt_label = dummy_label
  elif 'label' in graph_nodes[int(edge[1])].keys():
    tgt_label = graph_nodes[int(edge[1])]['label']
    serialized_nodes.add(int(edge[1]))
  else:
    LOGGER.warning("Unlabeled nodes in graph data for graph %s." % graph.name)
    tgt_label = "UNKNOWN_LABEL"
    serialized_nodes.add(int(edge[1]))

  return f'e {edge[0]} {edge[1]} {label} {src_label} {tgt_label}'


################# END edgeL serialization #################################################

####################### BEGIN: edgeL (edge list) parsing ###################

def parse_graphs_single_string(graphs_string: str, is_directed:bool = True, version: int = AUTO_VERSION, synthetic_dataset=False) -> List[Tuple[bool, nx.Graph]]:
  # split 
  graph_strings = _split_string(graphs_string)
  
  # parse graphs
  parsed_graphs = [parse_graph(serialized_graph, 
                               directed=is_directed, 
                               version=version, synthetic_dataset=synthetic_dataset,
                               parse_labels_json=False) for serialized_graph in graph_strings]
    
  # Remove "Nones"
  #parsed_graphs = [parsed_graph for parsed_graph in parsed_graphs if parsed_graph is not None]
    
  return parsed_graphs
  
def _split_string(mutliple_graphs_string: str) -> List[str]:
    # Separate header-wise
    regex = r"^t # (\S)*\n(?:e.*(?:\n|$))+"  # regex that looks for headers (t # foo\n) followed by an arbitrary number of edges (e something\n)
    matches = re.finditer(regex, mutliple_graphs_string, re.MULTILINE)
    return [match.group(0) for match in matches]
  
def parse_graphs(graph_strings: List[str], is_directed:bool = True, version: int = AUTO_VERSION, synthetic_dataset=False) -> List[Tuple[bool, nx.Graph]]:
  # parse graphs
  parsed_graphs = [parse_graph(serialized_graph, 
                               directed=is_directed, 
                               version=version, synthetic_dataset=synthetic_dataset,
                               parse_labels_json=False) for serialized_graph in graph_strings]
    
  # Remove "Nones"
  #parsed_graphs = [parsed_graph for parsed_graph in parsed_graphs if parsed_graph is not None]
    
  return parsed_graphs

def parse_graph(graph_string: str, synthetic_dataset: bool =False, directed: bool = True, version: int = AUTO_VERSION,
               parse_labels_json: bool = False, reduce_labels: bool = False, serialized_ids: Set[int] = None) -> Union[Tuple[bool, nx.Graph], Tuple[bool, None]]:
  ''' 
  Parses a graph in the form of a list of edges separated by a new line symbol. Each edge has a label and the id of source and target node as well as source and target node labels.
  Every graph starts with a header that includes the id or name of the graph. Example:
  t # 0
  e 0 1 c A B
  e 0 2 b A C
  e 1 2 a B C

  Since node labels are redundant, the consistency has to be checked, if a node appears in multiple edges.
  
  This method also supports adding ids of serialized nodes to the labels (to ensure that they are matched in a graph matching.)
  To enable this, a set with the corresponding node id's has to be given as "serialized_ids".
  
  returns True, Graph if the graph could be parsed correctly
  '''
  if directed:
    G = nx.DiGraph()
  else:
    LOGGER.error("Only DiGraphs supported currently. Use directed=True.")
    raise Exception("Only DiGraphs supported currently.")
  
  # t # graph_name/id
  regex_header = r"t # (.*)"

  lines = graph_string.split('\n')
  matches_header = re.match(regex_header, lines[0])
  
  if not matches_header:
    return False, None
  
  G.name = matches_header.group(1)

  is_correct=True
  for line in lines[1:]:
    add_edge = True
    if line == "$$":
      continue
    if len(line) == 0:
      break
    correct, src_id, tgt_id, edge_label, src_label, tgt_label = parse_edge(line, version=version, parse_labels_json=parse_labels_json, reduce_labels=reduce_labels, serialized_ids=serialized_ids)
        
    if not correct:
      is_correct= False
      add_edge =False
      if not synthetic_dataset:
        LOGGER.warning(f"Incorrect format. Couldn't parse edge: {line}")

    # add source node if not available
    if src_id in G.nodes:
      # verify consistency
      if src_label not in DUMMY_NODE_LABELS and not G.nodes(data=True)[src_id]['label'] == src_label:

        if not synthetic_dataset:
          LOGGER.warning(f"Nodes labels not consistent {G.nodes(data=True)[src_id]['label']} and {src_label}")
          return False, None
        else:
          is_correct = False
          add_edge = False

    elif add_edge:
      # add node
      G.add_node(src_id, label=src_label)


    # add target node if not available
    if tgt_id in G.nodes:
      # verify consistency
      if tgt_label not in DUMMY_NODE_LABELS and not G.nodes(data=True)[tgt_id]['label'] == tgt_label:
       if not synthetic_dataset:
         LOGGER.warning(f"Nodes labels not consistent {G.nodes(data=True)[tgt_id]['label']} and {tgt_label}")
         return False, None
       else:
          is_correct = False
          add_edge= False
    elif add_edge:
      # add node
      G.add_node(tgt_id, label=tgt_label)

    # add edge
    if add_edge:
      G.add_edge(src_id, tgt_id, label=edge_label)
  return is_correct, G
 
def is_header(input: str):
  regex_header = r"t # (.*)"

  matches_header = re.match(regex_header, input)
  
  if not matches_header:
    return False 
  return True

def parse_edge(edge_string: str, version: int=AUTO_VERSION, parse_labels_json=False, reduce_labels=False, serialized_ids: Set[int] = None):
  regex_edge_original = r"e (\d+) (\d+) (.+) (.+) (.+)"
  regex_edge_json = r"e (\d+) (\d+) (\"?\{.+\}\"?) (\"?\{.*\}\"?|_) (\"?\{.*\}\"?|_)"

  # Auto parse
  if version == AUTO_VERSION:
    matches_edge = re.match(regex_edge_json, edge_string)
    version = V_JSON
    if not matches_edge:
      matches_edge = re.match(regex_edge_original, edge_string)
      version == V_ORIGINAL
  # e src_id tgt_id edge_label src_label tgt_label
  elif version == V_ORIGINAL:
    matches_edge = re.match(regex_edge_original, edge_string)
  elif version == V_JSON:
    matches_edge = re.match(regex_edge_json, edge_string)
  else:
    LOGGER.warning(f"Version not supported: {version}")
    return False, None, None, None, None, None
  
  if not matches_edge:
    return False, None, None, None, None, None
  
  src_id = int(matches_edge.group(1))
  tgt_id = int(matches_edge.group(2))

  edge_label = str(matches_edge.group(3))
  src_node_label = str(matches_edge.group(4))
  tgt_node_label = str(matches_edge.group(5))
  
  # Special handling for V_JSON version of serialization
  if version == V_JSON:
    #underscores are mapped to {} for easier handling
    if src_node_label == "_":
      src_node_label = "\"{}\""
    if tgt_node_label == "_":
      tgt_node_label = "\"{}\""
    

    # Special handling in case node ids have to be added to the node labels:
    src_add_attributes = dict()
    tgt_add_attributes = dict() 
    if serialized_ids is not None and len(serialized_ids) > 0 and (src_id in serialized_ids or tgt_id in serialized_ids):
      if src_id in serialized_ids:
        src_add_attributes['serialized_node_id'] = src_id
      
      if tgt_id in serialized_ids:
        tgt_add_attributes['serialized_node_id'] = tgt_id     
    
    # Transform the edge and node labels to valid json (due to historical reason, the V_JSON is not valid json yet)  
    if parse_labels_json:
      edge_label = ChangeGraphEdge.to_json(edge_label, reduce=reduce_labels)
      if not src_node_label in DUMMY_NODE_LABELS:
        src_node_label = ChangeGraphNode.to_json(src_node_label, reduce=reduce_labels, add_fields=src_add_attributes)
      else:
        src_node_label = "{}"
      if not tgt_node_label in DUMMY_NODE_LABELS:
        tgt_node_label = ChangeGraphNode.to_json(tgt_node_label, reduce=reduce_labels, add_fields=tgt_add_attributes)
      else:
        tgt_node_label = "{}"
        
  elif version == V_ORIGINAL:
    # Special handling in case node ids have to be added to the node labels:
    if serialized_ids is not None and len(serialized_ids) > 0 and (src_id in serialized_ids or tgt_id in serialized_ids):
      if src_id in serialized_ids:
        src_node_label = str(src_id) + "_" + src_node_label
      if tgt_id in serialized_ids:
        tgt_node_label = str(tgt_id) + "_" + tgt_node_label
  
  return True, src_id, tgt_id, edge_label, src_node_label, tgt_node_label


def parse_lm_format_graph(graph_string):
  '''
  For training the language model and reading out motifs, we throw away some unnecessary information in the edgeL format.
  We correct for this here and parse this format.

  returns syntax_correct, corrected_syntax, parsed_graph. Syntax correct is True, if the graph is correct syntax, corrected_syntax contains the possibly parsable string. parsed_graph contains the networkx graph.
  '''
  corrected_syntax_graph = "t # 0\n" + graph_string[:graph_string.rindex('\n')+1]# We have to add header and cut off the additional characters (end of graph and new edge) here
  correct_syntax, parsed_pattern = parse_graph(corrected_syntax_graph) 
  return correct_syntax, corrected_syntax_graph, parsed_pattern

def get_prompt_graphs(prompt: str, seperator: str ="$$\n---\n", synthetic_dataset: bool =False) -> List[str]:
  """
  Splits the given prompt by the given "graph seperator" and corrects the last graph in the list.

  Args:
      prompt (str): A list of serialized graphs for language model based completion.
      seperator (str, optional): The seperator between serialized (partial) graphs.. Defaults to "$$\n---\n".

  Returns:
      List[str]: A list of serialized (partial) graphs.
  """
  
  # Split the prompt (which probably has more than one graph)
  partial_graphs = prompt.split(seperator)
  # Last graph has some additional beginning new line and "e" prompt, which we cut of
  if not synthetic_dataset:
    partial_graphs[-1] = partial_graphs[-1][:-2] # cut off new edge prompt
  
  return partial_graphs

def get_used_node_ids(graph_serialization: str, version=V_ORIGINAL) -> Set[int]:
  """
  Determines node ids used in a serialized (partial) graph.

  Args:
      graph_serialization (str): The serialized (partial) graph.

  Returns:
     Set(int): A set of used node ids.
  """
  used_node_ids = set()
  for line in graph_serialization.split('\n'):
      is_edge, src_id, tgt_id, edge_label, src_label, tgt_label = parse_edge(line, version=version)
      if is_edge:
          used_node_ids = used_node_ids.union({src_id, tgt_id})
  return used_node_ids

def get_prompt_only_nodes(prompt: str, completion: str, version=V_ORIGINAL) -> Set[int]:
  """For a (context) graph (serialized) and a possible completion for that serialized graph, this method determines
  the nodes that are only part of the prompt. but not of the completion.

  Args:
      prompt (str): The serialized context graph.
      completion (str): The serialized completion (candidate).

  Returns:
      Set[int]: The set of node ids of nodes that are only part of the prompt, but not part of the completion.
  """
  return  get_used_node_ids(prompt, version) - get_used_node_ids(completion, version)


def get_anchor_node_ids(prompt: str, completion: str, version=V_ORIGINAL) -> Set[int]:
  """
  For a (context) graph (serialized) and a possible completion for that serialized graph, this method determines
  the anchor nodes (by their ids), i.e., the nodes in the context graph/prompt that the completion is attached to.
  
  This can be used to reduce graph comparison to the graph completions, by ensuring that the anchor nodes are correctly matched.

  Args:
      prompt (str): The serialized context graph.
      completion (str): The serialized completion (candidate).

  Returns:
      Set(int): A set of anchor node ids.
  """
  return get_used_node_ids(completion, version).intersection(get_used_node_ids(prompt, version))

def get_anchored_completion_graph(prompt: str, completion: str,synthetic_dataset: bool= False, version=V_ORIGINAL) ->  Union[Tuple[nx.Graph, bool] , None]:
  """
  
  Parses the graph given by the prompt (context) and the completion but returns only the nodes of the completion. 
  To ensure also correct "glueing" to the context, the ids of the context graph are added to the label.

  Args:
      prompt (str): The serialized prompt (partial) graph (or context graph).
      completion (str): A serialized completion candidate for the prompt graph.
      version (int, optional): The version of the edgel format used for prompt and completion. Defaults to V_ORIGINAL.

  Returns:
      nx.Graph: A networkx graph for the completion.
  """
  
  # Firstly, we need a way do determine "new nodes" and "anchor nodes"
  anchor_node_ids = get_anchor_node_ids(prompt, completion, version=version)
  prompt_only_nodes = get_prompt_only_nodes(prompt, completion, version=version)
  
  full_graph = prompt + "\n" + completion

  # Secondly, we need a way to augment the labels for anchor nodes to ensure that anchor nodes are matched correctly
  # This functionality is available in the parse graph function via the attribute "serialized_ids"
  # Thirdly, we "reduce" the labels. Basically, attributes are very unlikely to be 100% correct. We compare these manually. 
  # Automatically, we can check structural correctness and type correctness. We therefore reduce the label to only carry type information.
  # This "reduction" is also available as a feature in the "parse_graph" function.
  correct, graph = parse_graph(full_graph, synthetic_dataset, directed=True, version=version, parse_labels_json=True, reduce_labels=True, serialized_ids=anchor_node_ids)
  if not correct and not synthetic_dataset:
    LOGGER.warning(f"Incorrect graph completion {completion}.")

    return None

  # Fourthly, after parsing the entire graph, we remove all nodes that are only part of the prompt (i.e., not part of the completion).
  graph.remove_nodes_from(prompt_only_nodes)
  return graph, correct
  
################# END edgeL parsing #################################################