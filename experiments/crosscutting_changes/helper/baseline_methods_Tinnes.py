from abc import ABC, abstractmethod
import ast
import json
import os
import re
import sys
import networkx as nx
from typing import Any, Callable, List, Set, Tuple, Union
from sentence_transformers import SentenceTransformer, util
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)

sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)

from experiments.crosscutting_changes.helper.HELPER_node_lables import compute_change_class, mark_neightbors
# FORMAT VERSION
V_ORIGINAL = 1 # Supports only string without whitespace characters for edge and node labels
V_JSON = 2 # Supports JSON labels for edges and nodes. Furthermore, "_" labels can be used to not repeat any label
AUTO_VERSION = 3 # Checks automatically first if it matches JSON then fallbacks to V_ORIGINAL

CHAT_MODEL_INSTRUCTION_TINNES_ADJUSTED = """
You are an assistant that is given a list of change graphs in an edge format. That is, the graph is given edge by edge. The graphs are directed, labeled graphs. An edge is serialized as
"e src_id tgt_id edge_label src_label tgt_label"

Labels are dictionaries. If a node appears in more than one edge, the second time it appears it is replaced by "_" to avoid repetition. 

E.g.:
e 0 1 a b bar
e 1 2 bla _ foo

The second edge here would be equivalent to:
"e 1 2 bla bar foo" 
The graph given is not yet complete. Exactly one edge is missing. Your task is to complete the graph by guessing the last edge. Give this missing edge in the format e src_id tgt_id edge_label src_label tgt_label
Note that the beginning "e" is already part of the prompt. Please do your reasoning internally and for efficieny just repeat the header of the graph of form "t # graph_name \n" again and than guess the ONE edge in edglL format, NOT how you got there."""

# TODO include if DB there There are some change graphs given as examples. Graphs are separated by "\n\n$$\n---\n".

#The last graph in this list of graphs is not yet complete. Exactly one edge is missing. 
#Your task is it, to complete the last graph by guessing the last edge. You can guess this typically by looking at the examples and trying to deduce the patterns in the examples. Give this missing edge in the format
#"e src_id tgt_id edge_label src_label tgt_label".
CHAT_MODEL_INSTRUCTION_TINNES = """
You are an assistant that is given a list of change graphs in an edge format. That is, the graph is given edge by edge. The graphs are directed, labeled graphs. An edge is serialized as
"e src_id tgt_id edge_label src_label tgt_label"

Labels are dictionaries. If a node appears in more than one edge, the second time it appears it is replaced by "_" to avoid repetition. 

E.g.:
e 0 1 a b bar
e 1 2 bla _ foo

The second edge here would be equivalent to:
"e 1 2 bla bar foo"

There are some change graphs given as examples. Graphs are separated by "\n\n$$\n---\n".

The last graph in this list of graphs is not yet complete. Exactly one edge is missing. 
Your task is it, to complete the last graph by guessing the last edge. You can guess this typically by looking at the examples and trying to deduce the patterns in the examples. Give this missing edge in the format
"e src_id tgt_id edge_label src_label tgt_label". Note that the beginning "e" is already part of the prompt. Please do your reasoning internally and for efficieny just repeat the header of the graph of form "t # graph_name \n" again and than guess the ONE edge in edglL format, NOT how you got there."""


#TODO only nodes please
PREDICTIONS_BASE_INSTRUCTION_OURS = """You are an assistant that is given a software model represented by a graph in json format.
Exactly one element node is missing with its one edge to the rest of the model are also missing. 
The edge starts from the FocusNode given below to the new one node. 
Your task is to complete the software model by predicting THIS ONE missing node (and its edges to the rest of the software model)
Give this ONE element ONLY in the format nodes:[...], edges:[...], where each node and edge is specified 
according to the commonly known json format specifing the id and all its attributes in a dictonary, for reference please have a look at this short example:
{
  "nodes": [
    {
      "id": "3",
      "label": {
          "type": "object",
          "className": "EStringToStringMapEntry",
          "attributes": {
            "id": "_XNoEVN6tEei97MD7GK1RmA",
            "key": "documentation",
            "value": "<p>\\r\\nProvisional for 4.3.\\r\\n</p>\\r\\n@noreference\\r\\n@since 1.0"
          }
        }
    } // where "id" specifies the node id, and the following label should contain all further detail e.g. eClass, and type ...
  ],
  "edges": [
    {
      "source": "2", 
      "target": "3", 
      "label": "{'changeType': 'DELETE', 'type': 'reference', 'referenceTypeName': 'details'}"
    }, ...
  ]
}

There are some change graphs given as examples. Graphs are separated by "\n---\n".
The last graph in this list of graphs is not yet complete.
Please do your reasoning internally and for efficieny just give the ONE node and ONE edge in json format, NOT how you got there.
The new node and its edge are modified elements, so they need to be marked either as changed, added or deleted.
Here are the few shot examples and the software model you are supposed to complete:
"""

CHAT_MODEL_INSTRUCTION_OURS = """You are an assistant that completes *exactly one* missing element (node OR edge) in a software model represented as a graph in JSON format. The model that needs completion will be appended after this instruction.

STRICT RULES (must follow exactly)
1. Output MUST be a single, valid JSON object and NOTHING ELSE. Do not add explanations, Markdown, comments, or any text outside the JSON. The JSON must be parseable by `json.loads()` (double quotes, no trailing commas).
2. Always include both top-level keys: `"nodes"` and `"links"`.
   - If the missing element is a **node**, return:
     - `"nodes": [ <one node object> ]`   ← exactly one node object
     - `"links": [ <zero-or-more edge objects> ]` ← list all edges that connect the new node to existing nodes in the model
   - If the missing element is an **edge**, return:
     - `"nodes": []`  ← empty list
     - `"links": [ <one edge object> ]`  ← exactly one edge object
3. **Node object schema** (flattened attributes):
   - Must contain `"id"` (string). Example: `"id": "42"`.
   - Include any other attributes as additional key/value pairs **on the same object** (no nested `"attributes"`).  
   - **All attribute values MUST be strings.**  
     - Boolean values should be the strings `"True"` or `"False"`.  
     - Use `"None"` for null/unknown values.
   - Example node object:
     ```json
     {
       "id": "3",
       "visibility": "PUBLIC_LITERAL",
       "qualifiedName": "model::Social Network Manager",
       "name": "Social Network Manager",
       "isIndirectlyInstantiated": "True",
       "isActive": "False",
       "eClass": "Component",
       "isfocusedNode": "False"
     }
     ```
4. **Edge object schema**:
   - Must contain `"source"` (string) and `"target"` (string) — these must be node ids present in the supplied model (use existing ids exactly).
   - `"key"` is **optional**. Include `"key"` (string) only if the dataset is a multigraph and you know the correct key; otherwise omit it.
   - Additional edge attributes are allowed; **all values must be strings** (booleans → `"True"`/`"False"`, missing → `"None"`).
   - Example edge object:
     ```json
     {
       "source": "161",
       "target": "200",
       "key": "0"
     }
     ```
5. **Do not add any other nodes or edges** beyond the one missing element (and the missing node's incident edges if the missing element is a node).
6. **Do not invent node ids that conflict** with the model's existing ids. Prefer using the original id if it is known from context. If you must create an id, choose a string id that does not collide with existing ids and document nothing (but prefer not to invent).
7. If any attribute values are unknown, use the string `"None"`. If an attribute is boolean, use `"True"` or `"False"` as strings.

EXAMPLES (must follow these JSON-only formats):

- Example for a missing **node** (single node + its incident edges):
```json
{
  "nodes": [
    {
      "id": "3",
      "visibility": "PUBLIC_LITERAL",
      "qualifiedName": "model::Social Network Manager",
      "name": "Social Network Manager",
      "isActive": "False",
      "eClass": "Component",
      "isfocusedNode": "False"
    }
  ],
  "links": [
    {"source": "3", "target": "10"},
    {"source": "3", "target": "25", "key": "0"}
  ]
}

- Example for a missing **edge**:
{
  "nodes": [],
  "links": [
    {"source": "161", "target": "200"}
  ]
}

IMPORTANT: return only the JSON (no surrounding triple backticks, no commentary). The model you will complete is given below — predict exactly one missing element in the requested JSON format.
"""

def connected_components(graph: nx.Graph) -> List[nx.Graph]:
    if nx.is_directed(graph):
        components = [graph.subgraph(c).copy() for c in nx.weakly_connected_components(graph)]
    else:
        components = [graph.subgraph(c).copy() for c in nx.connected_components(graph)]

    connected_component_id =0
    for c in components:
        c.diff_id = graph.diff_id
        #c.folder_name= graph.folder_name
        c.component_id = connected_component_id
        connected_component_id+=1


    return components

# new from us according to the paper they compute simple change graphs
def subgraph_with_connected_edges(graph: nx.Graph, node_tuples):
    # Extract the node IDs (they’re strings in your example)
    node_ids = {n for n, _ in node_tuples}

    # Collect all nodes that are either in node_ids or connected to them
    included_nodes = set(node_ids)
    included_edges = []
    for u, v , data in graph.edges(data=True):
        if u in node_ids or v in node_ids:
            included_nodes.add(u)
            included_nodes.add(v)
            included_edges.append((u, v, data))

    # Create the induced subgraph

    sub = graph.__class__()  # preserves DiGraph vs Graph etc.
    for n in included_nodes:
        sub.add_node(n, **graph.nodes[n])  # copy node attributes
    sub.add_edges_from(included_edges)

    sub.diff_id= graph.diff_id
    return sub

def get_component_of_node(graph: nx.Graph, node):
    #aufteilung nach changed items rein 
    graph = mark_neightbors(graph)
    change, nonchanged = compute_change_class(graph, withEmbeddings=False)

    # because the focus node is usually preserved, specially at the begging
    data = next((d for n, d in nonchanged if str(n) == str(node)), None)
    if data is not None:
        change.append((str(node), data))
    #list(graph.nodes(data=True))[0]
    subgraph = subgraph_with_connected_edges(graph, change)
    components = connected_components(subgraph)
    for c in components:
        if str(node) in c.nodes:
            if (c.size()<= 20 and len(c.edges)> 0): 
              return c  # return the component (as a subgraph)
            
    component = nx.ego_graph(graph, str(node), radius=1)
    
    if len(component.edges)== 0:
       return subgraph
    component.diff_id = graph.diff_id
    return component  # node not found

# dfs-serialization
def dfs_edges(graph: nx.Graph):
    # Get all nodes with in-degree zero (we need to start the dfs from there)
    # roots = [node for (node, val) in graph.in_degree() if val == 0]
    return graph, [(x, y, graph.get_edge_data(x, y)) for (x, y, dir) in
                   nx.algorithms.traversal.edgedfs.edge_dfs(graph, orientation='ignore')]  # , source=roots))


def serialize_graph(graph: nx.Graph, serialization_strategy: Callable[[nx.Graph],Tuple[nx.Graph, List[Tuple]]] = dfs_edges, is_completion: bool=False, serialized_nodes=None, version=V_ORIGINAL) -> Tuple[str, Set[int]]:
  if serialized_nodes is None: # This is to avoid confusion if serialized_nodes, by default, is set to set(), then calling this method multiple times will keep the state, which leads to unexpected behaviour.
    serialized_nodes = set()
  graph_string = ''
  # Add header
  if not is_completion:
    graph_string += f't # {graph.diff_id}\n'
  # Serialize edges
  graph, edges = serialization_strategy(graph)
  for edge in edges:
    graph_string+=serialize_edge(edge, graph, serialized_nodes, version=version)+'\n'
  
  return graph_string, serialized_nodes



def serialize_edge(edge: Tuple, graph: nx.Graph, serialized_nodes=set(), version=V_ORIGINAL):
  if 'label' not in edge[2].keys():
    print("Unlabeled edges in graph data for graph %s." % graph.name)
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
    print(f"Unknown EdgL version: {version}")
    dummy_label = "_"

  if int(edge[0]) in serialized_nodes:
    src_label = dummy_label
  elif 'label' in graph_nodes[int(edge[0])].keys():
    src_label = graph_nodes[int(edge[0])]['label']
    serialized_nodes.add(int(edge[0]))
  else:
    print("Unlabeled nodes in graph data for graph %s." % graph.name)
    src_label = "UNKNOWN_LABEL"
    serialized_nodes.add(int(edge[0]))
    
  if int(edge[1]) in serialized_nodes:
    tgt_label = dummy_label
  elif 'label' in graph_nodes[int(edge[1])].keys():
    tgt_label = graph_nodes[int(edge[1])]['label']
    serialized_nodes.add(int(edge[1]))
  else:
    print("Unlabeled nodes in graph data for graph %s." % graph.name)
    tgt_label = "UNKNOWN_LABEL"
    serialized_nodes.add(int(edge[1]))

  return f'e {edge[0]} {edge[1]} {label} {src_label} {tgt_label}'





DUMMY_NODE_LABELS = ["_", "\"{}\"", "{}"]
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
    print("Only DiGraphs supported currently. Use directed=True.")
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
        print(f"Incorrect format. Couldn't parse edge: {line}")

    # add source node if not available
    if src_id in G.nodes:
      # verify consistency
      if src_label not in DUMMY_NODE_LABELS and not G.nodes(data=True)[src_id]['label'] == src_label:

        if not synthetic_dataset:
          print(f"Nodes labels not consistent {G.nodes(data=True)[src_id]['label']} and {src_label}")
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
         print(f"Nodes labels not consistent {G.nodes(data=True)[tgt_id]['label']} and {tgt_label}")
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
    print(f"Version not supported: {version}")
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


def clean_up_string(input: str) -> str:
    input = input.replace('\\\'', '"') # used for quotes in strings
    input = re.sub('\'\'([^\']+)\'\'', '"\\1"', input) # double single quotes also used for quotes in strings
    #input = re.sub('\'([\':,]*)\'(?![,\]\}\:])', '\\1', input)
    input = re.sub('\s\'([\w\s\.,-]*)\'[^,\]}:]', '\\1', input) # word or strings surrounded by quotes and whitespaces (or other ending characters not including default json stuff such as brackets, colons or commata).
    input = re.sub('\'(\\w+\(\))\'', '\\1', input) # sometimes method names are put in quotes
    
    
    input = re.sub('\'(\\w+)\'\\\\\\\\nVersion .', '\\1', input) # one-off thing, don't know how this string actuall is created but it's there, so we have to handle it.
    input = re.sub('\'(\\w+)\'\\\\nVersion .', '\\1', input) # one-off thing, don't know how this string actuall is created but it's there, so we have to handle it.
    input = re.sub('\'\\\\\\\\nVersion .', '', input) # one-off thing, don't know how this string actuall is created but it's there, so we have to handle it.
    input = re.sub('\'s ', 's ', input) # one-off thing
    
    input = re.sub('\'stack\' ', 'stack ', input) # one-off thing
    input = re.sub(' \'instructions\' ', ' instructions ', input) # one-off thing
    input = re.sub('\'MIME: \'', 'MIME:', input) # one-off thing
    input = re.sub('\'MIME:', 'MIME:', input) # one-off thing
    input = re.sub('\'selected\' ', 'selected ', input) # one-off thing
    input = re.sub(': \'ecore::EDoubleObject\'', ': ecore::EDoubleObject', input) # one-off thing
    input = re.sub('\'in\' ', 'in ', input) # one-off thing
    input = re.sub('_\'in\'', '_in', input) # one-off thing
    input = re.sub('\s\'in\'', 'in', input) # 'in', 'inout', 'out', pr 'return'
    input = re.sub('\s\'inout\'', 'inout', input) # 'in', 'inout', 'out', pr 'return'
    input = re.sub('\s\'out\'', 'out', input) # 'in', 'inout', 'out', pr 'return'
    input = re.sub('\'out_', 'out_', input) # 'in', 'inout', 'out', pr 'return'

    
    input = re.sub('\s\'return\'', 'return', input) # 'in', 'inout', 'out', pr 'return'
    input = re.sub('_\'context\'', '_context', input) # one-off _context
    input = re.sub('\'\*\'', '\*', input) # one-off '*'
    input = re.sub(' \'alt\'', ' alt', input) # one-off 'alt'
    input = re.sub('_\'conte', '_conte', input) # one-off _'conte

    input = re.sub('_\'body\'', '_body', input) # one-off _'body'
    input = re.sub('\._\'', '._', input) # one-off ._'

    input = re.sub('\'::\'', '::', input) # 'in', 'inout', 'out', pr 'return'
    input = re.sub('\'create\'(?![,\]\}\:])', 'create', input) # 'in', 'inout', 'out', pr 'return'
    input = re.sub('\'ignore\'(?![,\]\}\:])', 'ignore', input) # 'in', 'inout', 'out', pr 'return'


    return input



  
  ################# END JSON Graph Label Parser #####################################


  
  ################# BEGIN cleanup for ast literal eval #####################################
  # same as above but some need to be removed , added 



class ChangeGraphElement():
  def __init__():
    pass
  
  @classmethod
  def from_string(cls, input: str):
    input = input.strip('"')
    input = clean_up_string(input)
    try:
      obj_dict = ast.literal_eval(input)
    except Exception as e:
      print(f"Object couldn't be parsed: {input}")
      raise e
    return cls(**obj_dict)

class ChangeGraphNode(ChangeGraphElement):
  def __init__(self, changeType: str=None, type: str=None, className: str=None, attributeName: str=None, attributes: Any = None, valueBefore: str = None, valueAfter: str = None):
    self.changeType = changeType
    self.type = type
    self.className = className
    self.attributeName = attributeName
    self.attributes = attributes
    self.valueBefore = valueBefore
    self.valueAfter = valueAfter
    
  def _to_json(self, reduce=True, additional_keep_fields: List[str] = []):
    change_graph_node = self
    if reduce:
      if self.changeType == "Change": # We have a Change Attribute Node
        change_graph_node = {"changeType": self.changeType, "className": self.className, "attributeName": self.attributeName} 
      else:
        change_graph_node = {"changeType": self.changeType, "className": self.className}

      for field in additional_keep_fields:
        change_graph_node[field] = getattr(self, field)
    else:
      #TODO obj to dict
      change_graph_node = {attr: getattr(self, attr) for attr in self.__dict__}
      pass
        
    return json.dumps(change_graph_node, sort_keys=True)
    
  @classmethod
  def to_json(cls, original_node_string: str, reduce=True, add_fields: ast.Dict=dict()) -> str:
    """
    
    Parse the input node string and extract changeType, type, className, and attributeName if applicable.
    This method also outputs a valid json, removes unneccesary quotes.

    Args:
        original_node_string (str): The original node label
        reduce (bool, Optional): True, if only specific values should be serialized.
        add_fields (Dict, Optional): A dictionary of attributes and values that should be added to the serialization.


    Returns:
        str: A probably reduced JSON node label, throwing away attribute specific information.
    """
    change_graph_node = cls.from_string(original_node_string)
    for key, value in add_fields.items():
      setattr(change_graph_node, key, value)

    return change_graph_node._to_json(reduce=reduce, additional_keep_fields=list(add_fields.keys()))
  

class ChangeGraphEdge(ChangeGraphElement):
  def __init__(self, changeType: str=None, type: str=None, referenceTypeName: str=None, attributes: Any=None):
    self.changeType = changeType
    self.type = type
    self.referenceTypeName = referenceTypeName
    self.attributes = attributes
    
  @classmethod
  def to_json(cls, original_edge_string: str, reduce=True) -> str:
    """
    
    Parse the input edge string and and transform it to valid json.
    This method also outputs a valid json, removes unneccesary quotes.

    Args:
        original_edge_string (str): The original edge label

    Returns:
        str: The JSON edge label
    """
    change_graph_edge = cls.from_string(original_edge_string)
    
    if reduce:
      if change_graph_edge.type == "attribute": # We have a Change Attribute Edge
        change_graph_edge = {"changeType": change_graph_edge.changeType, "type": change_graph_edge.type} 
      else:
        change_graph_edge = {"changeType": change_graph_edge.changeType, "referenceTypeName": change_graph_edge.referenceTypeName} 

    return json.dumps(change_graph_edge, sort_keys=True)


def calculate_structure_correctness(source_node, gt_data, gen_data):
    correct = False
    for link in gen_data["links"]:
       if link['source'] == str(source_node): 
          correct = True 

          #since this is the node already existing we remove it from gt
          for node in gen_data["nodes"]: 
            if node["id"] == str(source_node):
              gen_data["nodes"].remove(node)

       return correct
       
def calculate_change_type_correctness(gt_data, gen_data): 
    correct = False
    change_types = set()
    for entry in gt_data:
       for node in entry.get("nodes", []):
        change_types.add(node["changeType"])

    for gen in gen_data["nodes"]:
        if gen["changeType"] in change_types:
           correct=True


    return correct
def calculate_type_correctness(gt_data, gen_data): 
    correct = False 
  
    classNames = set()
    types = set()
    for entry in gt_data:
       for node in entry.get("nodes", []):
        types.add(node["type"])
        classNames.add(node["className"])
   
    for gen in gen_data["nodes"]:
        if gen["type"] in types and gen["className"] in classNames:
           correct=True

    return correct

class EvaluationMetric(ABC):
    """Abstract base class for evaluation metrics."""

    @abstractmethod
    def compute(self, ground_truth, prediction) -> float:
        """
        Compute the evaluation metric.

        Args:
            ground_truth: The ground truth data.
            prediction: The predicted data.

        Returns:
            float: The computed metric value.
        """
        pass


class CosineSimilarity(EvaluationMetric):
    """Concrete implementation of cosine similarity metric."""

    def __init__(self):
        super().__init__()  
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    def _preprocess_graph_json(self, graph_json: dict) -> str:
        
        """
        Convert validated graph JSON into a canonical string representation
        for semantic comparison.
        - Sorts nodes and links for consistency
        - Flattens into 'key=value' style lines
        """

        # Sort nodes by ID for consistency
        nodes = sorted(graph_json["nodes"], key=lambda n: n["id"])
        links = sorted(
            graph_json["links"],
            key=lambda e: (e["source"], e["target"], e.get("key", ""))
        )
        # Flatten nodes
        node_strings = []
        for node in nodes:
            attrs = [f"{k}={v}" for k, v in sorted(node.items())]
            node_strings.append("NODE(" + ", ".join(attrs) + ")")

        # Flatten links
        link_strings = []
        for link in links:
            attrs = [f"{k}={v}" for k, v in sorted(link.items())]
            link_strings.append("LINK(" + ", ".join(attrs) + ")")

        # Concatenate into one canonical string
        return "\n".join(node_strings + link_strings)
    def compute(self, ground_truth, prediction) -> float:
        # Preprocess graph JSONs into canonical strings such that semantic similarity is more accurately captured
        ground_truth_str = self._preprocess_graph_json(ground_truth)
        prediction_str = self._preprocess_graph_json(prediction)

        # Generate embeddings
        ground_truth_embedding = self.embedding_model.encode(ground_truth_str, convert_to_tensor=True)
        prediction_embedding = self.embedding_model.encode(prediction_str, convert_to_tensor=True)

        # Compute cosine similarity
        similarity = util.pytorch_cos_sim(ground_truth_embedding, prediction_embedding).item()
        return similarity
    


def transform_gt_entry(entry, source_node):
    """
    Converts a ground-truth element (from your predecessors list)
    into the same JSON-like format used by metric functions.
    """
    nodes = []
    links = []

    # --- parse node info ---
    if "node_data" in entry and "label" in entry["node_data"]:
        try:
            node_label = ast.literal_eval(entry["node_data"]["label"])  # parse the string dict
        except Exception:
            node_label = {}

        # flatten so it looks like your metric format expects
        node_obj = {
            "id": entry.get("successor", "unknown"),
            "className": node_label.get("className", "None"),
            "type":  node_label.get("type", "None"), 
            "changeType": node_label.get("changeType", "None")
        }
        nodes.append(node_obj)

    # --- parse edge info ---
    if "edges" in entry:
        for e in entry["edges"]:
            try:
                edge_label = ast.literal_eval(e["label"])
            except Exception:
                edge_label = {}
            
            link_obj = {"source": str(source_node), "target": str(entry.get("successor", "unknown")), **edge_label}
            links.append(link_obj)

    return {"nodes": nodes, "links": links}




def transform_gen_graph(graph, source_node):
    """
    Converts a NetworkX DiGraph or MultiDiGraph into
    the JSON-like structure expected by calculate_type_correctness().
    """
    nodes = []
    links = []

    # --- collect node info ---
    for n, data in graph.nodes(data=True):
        #if str(n) != source_node: 
          try:
              node_label = ast.literal_eval(data["label"])
              if isinstance(node_label, str):
                 node_label = ast.literal_eval(node_label) 
                 
              node_obj = {
              "id": str(n),
              "className": node_label.get("className", "None"),
              "type":  node_label.get("type", "None"), 
              "changeType": node_label.get("changeType", "None")
              }
              nodes.append(node_obj) # parse the string dict
          except Exception:
              node_label = {}

          

    # --- collect edge info ---
    if graph.is_multigraph():
        for u, v, key, data in graph.edges(keys=True, data=True):
            link_obj = {"source": str(u), "target": str(v), "key": str(key), **data}
            links.append(link_obj)
    else:
        for u, v, data in graph.edges(data=True):
            try:
                edge_label = ast.literal_eval(data["label"])
                if isinstance(edge_label, str):
                    edge_label = ast.literal_eval(edge_label) 
                edge_obj = {
                "type":  edge_label.get("type", "None"), 
                "changeType": edge_label.get("changeType", "None")
                }
                link_obj = {"source": str(u), "target": str(v), **edge_obj}
                links.append(link_obj)   # parse the string dict
            except Exception:
                edge_label = {}
            

    return {"nodes": nodes, "links": links}




def transform_gen_list(entry, source_node):
    """
    Converts a ground-truth element (from your predecessors list)
    into the same JSON-like format used by metric functions.
    """
    nodes = []
    links = []

    for n in entry["nodes"]: 
        try:
              node_label = ast.literal_eval(n["label"])  # parse the string dict
        except Exception:
              node_label = {}
       
        node_obj = {
              "id": n["id"],
              "className": node_label.get("className", "None"),
              "type":  node_label.get("type", "None"), 
              "changeType": node_label.get("changeType", "None")
          }
        nodes.append(node_obj)

    for e in entry["edges"]:
        try:
            edge_label = ast.literal_eval(e["label"])
        except Exception:
            edge_label = {}
        
        link_obj = {"source": e["source"], "target":  e["target"], **edge_label}
        links.append(link_obj)

    return {"nodes": nodes, "links": links}
    