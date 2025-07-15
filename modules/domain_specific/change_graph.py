import ast, json
import os
import random
import re
import shutil
import sys
from copy import deepcopy
from math import ceil, floor
from typing import Any, Dict, List
import networkx as nx
import logging

CHANGE_PREFIX={'Add_', "Del_", "Pres_", "Preserve_", "Remove_"}
PREFIX_DELETABLE =('Add', 'Delete', 'Remove', 'Change')

DUMMY_NODE_LABELS = ["_", "\"{}\"", "{}"] # Node labels used to avoid repetition in edge serializations

LOGGER = logging.getLogger("ChangeGraphHandling")

################ BEGIN JSON Graph Label Parser #####################################


class ChangeGraphTransformer:
  
  @classmethod
  def get_json_graph(cls, input: nx.Graph, reduce: bool=False, label_field='label'):
    
    for node in input.nodes(data=True):
      node_label = node[1][label_field]
      if node_label in DUMMY_NODE_LABELS:
        continue
      node[1][label_field] = ChangeGraphNode.to_json(node_label, reduce=reduce)
    
    for edge in input.edges(data=True):
      edge[2]['label'] = ChangeGraphEdge.to_json(edge[2][label_field], reduce=reduce)

    return input


# TODO move this to another module and try to better integrate with networkx
# TODO probably even bettern than working with json strings would be to work with the classes directly as data and defining __eq__ method.
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
      LOGGER.error(f"Object couldn't be parsed: {input}")
      raise e
    return cls(**obj_dict)

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
  def to_json(cls, original_node_string: str, reduce=True, add_fields: Dict=dict()) -> str:
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


def clean_up_string_ast_literal(input: str) -> str:
    input = input.replace('\\\'', '"') # used for quotes in strings

    #But your final format is for ast.literal_eval, which expects single quotes for strings (not JSON-style).
    #Worse, it may leave things like ''<p>... partially converted, causing invalid syntax like ''<p>\\r\\n....
    input = re.sub('\'\'([^\']+)\'\'', '"\\1"', input) # double single quotes also used for quotes in strings
    #input = re.sub('\'([\':,]*)\'(?![,\]\}\:])', '\\1', input)




     # word or strings surrounded by quotes and whitespaces (or other ending characters not including default json stuff such as brackets, colons or commata).
     # f you want to remove the quotes around certain strings without removing the character following the closing quote, you should modify the regex to use a lookahead assertion. This way, the regex will check for the condition without consuming the character, which means it won't be replaced or removed during the substitution.
    input = re.sub('\s\'([\w\s\.,-]*)\'(?=[^,\]}:])', '\\1', input)


    input = re.sub('\'(\\w+\(\))\'', '\\1', input) # sometimes method names are put in quotes
    
    
    input = re.sub('\'(\\w+)\'\\\\\\\\nVersion .', '\\1', input) # one-off thing, don't know how this string actuall is created but it's there, so we have to handle it.
    input = re.sub('\'(\\w+)\'\\\\nVersion .', '\\1', input) # one-off thing, don't know how this string actuall is created but it's there, so we have to handle it.
    input = re.sub('\'\\\\\\\\nVersion .', '', input) # one-off thing, don't know how this string actuall is created but it's there, so we have to handle it.
    input = re.sub('\'s ', 's ', input) # one-off thing
    input = re.sub(r"(?<=[a-zA-Z])'s", r"s", input)
    
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

    input = input.replace('\\r', '\\\\r')
    input = input.replace('\\n', '\\\\n')
    input = input.replace('\\t', '\\\\t') 

  
   # pattern_value = r"(value':\s*')(.*?)(\}+)"

    #def replacer(m):
        # m.group(1) => "value': "
        # m.group(2) => one or two single quotes right after value':
        # m.group(3) => the middle text (no single quotes, thanks to [^']*)
        # m.group(4) => one or two single quotes right before the braces
        # m.group(5) => the closing brace(s), e.g. "}}" or "}" etc.

        # Remove ALL single quotes from the middle text.
       # cleaned_middle = m.group(2).replace("'", "")

        # Reassemble. We unify the outer quotes to a single quote on each side.
        # So even if the original had two quotes (''), it now becomes one (').
      #  return f"{m.group(1)}{cleaned_middle}'{m.group(3)}"

    #input = re.sub(pattern_value, replacer, input)
    return input




############################ v1 -> v2 ######################################
# TODO ctinnes this is very messy, we should split the parsing of json labels, trafo of chagne graph v1 to v2 and generation of prompts etc.
def transform_token_node_attributes(token):
  change_type, obj_type = token.split('_')
  return {
    'changeType': change_type,
    'type': "object",
    'className': obj_type,
    'attributes': {
      'id': '',
      'name': '',
      'fqn': '',
      'maximumCountInCar': '',
      'coachAttributeValues': []
    }
  }
def replace_last_occur(text):
  return text.rsplit("}", 1)[0] + "\\}" + text.rsplit("}", 1)[1] if "}" in text else text

def transform_v1_to_v2(file_path_source, file_path_destination):

  all_transformed_data = []
  e=0

  with open(file_path_source, 'r') as f:
    transformed_data_one_example = {}
    for line in f:

        original_data = json.loads(line.strip())
        prompt_id = original_data["id"]

        for part in ["prompt","completion" , "completion_string" ]:
                data = original_data[part]
                edges = [line.strip() for line in data.split("\n") if line.strip()]
                if (edges[0].startswith("t")):
                  edges.pop(0)


                transformed_edges = []

                # Loop through the edges to transform them
                for edge in edges:
                      tokens = re.split(r'\s+', edge)

                      if(tokens==["$$"] ):
                        transformed_edges.append(tokens[0])
                      elif ( tokens==["---"] ):
                        transformed_edges.append(tokens[0])
                        transformed_edges.append("t # " + json.dumps(prompt_id))


                      elif (part=="completion_string" and not re.fullmatch(r'(e\s+)?(\d+) (\d+) (\w+) (\w+) (\w+)', edge)):
                        #usually at the very end, cpt produced too many edges
                        print("not syntactically correct completion string")


                      elif ( tokens!=["e"]):
                        if ("e" not in tokens ):
                          tokens.insert(0, "e")
                          e += 1
                          print("e for edge is missing")
                        edge_type =  tokens[3].split('_')
                        dict_edge = {}
                        dict_edge["changeType"] = edge_type[0]
                        dict_edge["type"] = "reference"
                        dict_edge["referenceTypeName"] = edge_type[1]

                        node_first_new_attribute = transform_token_node_attributes(tokens[4])
                        node_second_new_attribute = transform_token_node_attributes(tokens[5])

                        transformed_edge = 'e ' + tokens[1] + ' ' + tokens[2] + ' \"' + json.dumps(
                          dict_edge) +  '\" \"' + json.dumps(node_first_new_attribute) +  '\" \"' + json.dumps(
                          node_second_new_attribute) + '\"'

                        transformed_edges.append(transformed_edge)

                transformed_data_one_example[part] = "\n".join(transformed_edges)

        final_prompt={}
        final_prompt["sample_id"]=  prompt_id
        final_prompt["change_type"] ="None"
        final_prompt["detail_type"] = "None"
        final_prompt["context_type"] = "None"
        final_prompt["similar_few_shot"] = "None"
        final_prompt["comment"] = "None"
        final_prompt["id"] =  prompt_id
        final_prompt["prompt"]=   transformed_data_one_example["prompt"]
        final_prompt["completion"] =   transformed_data_one_example["completion"]
        final_prompt["completion_string"] = transformed_data_one_example["completion_string"]
        all_transformed_data.append(json.dumps(final_prompt))

    result = "\n".join(all_transformed_data)
    print(e)
    with open(file_path_destination, 'w') as f:
      f.write(result)


if __name__ == "__main__":
  """
  Executes when called as python module.
  """
  if len(sys.argv) == 3:
     transform_v1_to_v2(sys.argv[1], sys.argv[2])
  else:
    print("Unexpected number of arguments. Call like python eval_completions.py [input_path] [output_path].")





####################### GRAPH MODIFICATIONS ########################


#TODO aufräumen
def remove_random_edge(graph, number_of_edges_to_remove):
    G = deepcopy(graph)
   # G = graph.copy()
    edges = G.edges(data=True)
    # Filter edges based on label condition
    # filtered_edges = [(u, v) for u, v, data in edges if data.get('label', '').startswith(('Add_', 'Delete_'))]

    filtered_edges = []

    for u, v, data in edges:
        stri = data.get('label', '').strip('"').replace("'", '"')
        data_dict = json.loads(stri)
        if (data_dict.get('changeType').startswith(('Add', 'Delete', 'Remove', 'Change'))):

            stri_u = G.nodes[u].get('label', '')
            stri_v = G.nodes[v].get('label', '')
            _, temp_u = re.split(r"'changeType': |\"changeType\": ", stri_u)
            _, temp_v = re.split(r"'changeType': |\"changeType\": ", stri_v)
            temp_v = temp_v.replace('"', "'")
            temp_u = temp_u.replace('"', "'")
            if (data_dict.get('changeType').startswith(('Change'))):
              filtered_edges.append((u, v, data_dict.get('changeType') + "_attribute"))

            elif temp_u.startswith(("'Add'", "'Delete'", "'Remove'", "'Change'")) or temp_v.startswith(
                    ("'Add'", "'Delete'", "'Remove'", "'Change'")):


              filtered_edges.append((u, v, data_dict.get('changeType') + "_node"))
            else:  # both nodes connected to the edge are preserved


              filtered_edges.append((u, v, data_dict.get('changeType') + "_edge"))


    number_of_edges_to_remove = min(number_of_edges_to_remove, len(filtered_edges))
    number_of_edges_modified_graph = G.size() - number_of_edges_to_remove

    random_edges = random.sample(filtered_edges, number_of_edges_to_remove)

    nodes = []
    for e in random_edges:
        nodes.append((e[0], G.nodes[e[0]]))
        nodes.append((e[1], G.nodes[e[1]]))

    G.remove_edges_from(random_edges)

    completion = reconstruct_graph(G.name, G.diff_id , nodes=nodes,
                                   edges=((e[0], e[1], graph.get_edge_data(e[0], e[1])) for e in random_edges))
    change_types = [r[2]  for r in random_edges]
    return G, completion, number_of_edges_modified_graph, number_of_edges_to_remove, change_types


"""def get_changeType_type_from_dict(data_dict_str, type):
    stri = data_dict_str.get('label', '')

    if (type=="edge"):
        stri = stri.strip('"').replace("'", '"')
    if (type=="node"):
        stri = ChangeGraphNode.to_json(stri)

    data_dict = json.loads(stri)

    #if 'label' in data_dict_str:
     #   string = data_dict.split('_')
      #  changeType = string[0]
     #   type = string[1]
     #   return changeType, type

   # else:
    return data_dict.get('changeType'), data_dict.get('type')

    #TODO if this would not work, please refer to old lm2eo version
    # _, temp_u = re.split(r"'changeType': |\"changeType\": ", source_node_label)
    # _, temp_v = re.split(r"'changeType': |\"changeType\": ", target_node_label)
    # temp_v = temp_v.replace('"', "'")
    # temp_u = temp_u.replace('"', "'")
"""

"""def remove_random_edge(graph, number_of_edges_to_remove):
    G = graph.copy()
    print(G.name)
    edges = G.edges(data=True)

    # specifies which edges actually represent a natural completion
    filtered_edges = []

    for source, target, data in edges:

        changeType_edge, type_edge = get_changeType_type_from_dict(data, "edge")
        source_node_changeType, source_node_type = get_changeType_type_from_dict(G.nodes[source], "node")
        target_node_changeType, target_node_type = get_changeType_type_from_dict(G.nodes[target], "node")

        if (changeType_edge.startswith(('Change'))):

            filtered_edges.append((source, target, changeType_edge + "_attribute"))

        elif source_node_changeType.startswith(PREFIX_DELETABLE) or target_node_changeType.startswith(
                    PREFIX_DELETABLE):

            filtered_edges.append((source, target, changeType_edge  + "_node"))
        else:  # both nodes connected to the edge are preserved

            filtered_edges.append((source, target, changeType_edge  + "_edge"))


    number_of_edges_to_remove = min(number_of_edges_to_remove, len(filtered_edges))
    number_of_edges_modified_graph = G.size() - number_of_edges_to_remove

    random_edges = random.sample(filtered_edges, number_of_edges_to_remove)

    nodes = []
    for e in random_edges:
        nodes.append((e[0], G.nodes[e[0]]))
        nodes.append((e[1], G.nodes[e[1]]))

    G.remove_edges_from(random_edges)

    completion = reconstruct_graph(G.name, nodes=nodes,
                                   edges=((e[0], e[1], graph.get_edge_data(e[0], e[1])) for e in random_edges))
    change_types = [r[2]  for r in random_edges]
    return G, completion, number_of_edges_modified_graph, number_of_edges_to_remove, change_types
"""

def remove_random_nodes(graph, number_nodes_to_remove):
    G = deepcopy(graph)
    nodes = G.nodes(data=True)
    # Filter nodes based on label condition
    filtered_nodes = []

    for n, data in nodes:
        stri = data.get('label', '')
        _, _, temp = stri.partition("'changeType': ")
        if (temp.startswith(PREFIX_DELETABLE)):
            changeType = temp.split("'")[1]
            filtered_nodes.append((n, data, changeType))

    number_nodes_to_remove = min(number_nodes_to_remove, len(filtered_nodes))

    # Select x random nodes
    random_nodes = random.sample(filtered_nodes, number_nodes_to_remove)
    random_nodes_all = random_nodes.copy()

    edges_data_list = []
    for node in random_nodes:
        # Get the edges connected to the node
        neighbors = set(G.predecessors(node[0])).union(set(G.successors(node[0])))
        for neighbor in neighbors:
            if G.has_edge(node[0], neighbor):
                edge_data = G.get_edge_data(node[0], neighbor)
                edge = (node[0], neighbor, edge_data)
                edges_data_list.append(edge)

            if G.has_edge(neighbor, node[0]):
                edge_data = G.get_edge_data(neighbor, node[0])
                edge = (neighbor, node[0], edge_data)
                edges_data_list.append(edge)

        random_nodes_all.append((neighbor, dict(G.nodes[neighbor])))

    number_nodes_prompt = G.size() - len(edges_data_list)
    for n in random_nodes:
        G.remove_node(n[0])

    change_type = [r[2] + "_node" for r in random_nodes]

    completion = reconstruct_graph(G.name,G.diff_id,  nodes=random_nodes_all, edges=edges_data_list)

    return G, completion, number_nodes_prompt, number_nodes_to_remove, change_type


#TODO dynmatisch anpassbar machen
def modify_graphs(graph_components):
    modified_graphs = []
    completions = []
    count_removals = {}
    change_types = {}
    count_modified_graph_edges = {}

    for graph in graph_components:
        modified_graph, completion, modified_graph_edges, removals, change_type = remove_random_edge(graph, 1)  # number of esges to remove =1
        # modified_graph, completion,  removals, change_type = remove_random_nodes(graph,2)
        modified_graphs.append(modified_graph)
        completions.append(completion)
        count_removals[modified_graph.name] = removals
        change_types[modified_graph.name] = change_type
        count_modified_graph_edges[modified_graph.name] = modified_graph_edges

    return modified_graphs, completions, count_modified_graph_edges, count_removals, change_types

def save_prompt_completion(prompt, completion, file_path):
    with open(file_path, 'w') as f:
        f.write(prompt + '\n$$\n' + completion)

# deprecated method to create training samples by random cuts in the edgel serialization, see modify graphs for the new approach
def read_cut_write(input_folder, output_folder, lower_cut_percentage, upper_cut_percentage):
    # Create the output folder first (if it doesn't exist yet)
    shutil.rmtree(output_folder, ignore_errors=True)
    os.makedirs(output_folder, exist_ok=False)

    for file in os.listdir(input_folder):
        with open(input_folder + '/' + file, 'r') as f:
            graph = f.read()
            # We can throw away the header here
            graph_lines = list(filter(None, graph.split('\n')))[1:]

            nb_edges = len(graph_lines)
            graph = '\n'.join(graph_lines)

            # We want at least one edge in the prompt and one in the completion. This is not possible with one edge, so we skip.
            if nb_edges == 1:
                LOGGER.info("There is a graph with only one edge. Ommiting.")
                continue

            # LOGGER.info(f'{file}, {nb_edges}, {ceil(lower_cut_percentage * nb_edges)}, {floor(upper_cut_percentage * nb_edges)} ')

            lower_cut_point = random.randint(0, ceil(lower_cut_percentage * nb_edges))
            middle_cut_point = random.randint(floor(lower_cut_percentage * nb_edges),
                                              ceil(upper_cut_percentage * nb_edges))
            upper_cut_point = random.randint(floor(upper_cut_percentage * nb_edges), nb_edges)

            # At least one edge in completion and prompt
            lower_cut_point = min(nb_edges - 1, max(1, lower_cut_point))
            middle_cut_point = min(nb_edges - 1, max(1, middle_cut_point))
            upper_cut_point = min(nb_edges - 1, max(1, upper_cut_point))

            # Cut and save
            prompt, completion = cut_graph(graph, lower_cut_point)
            output_file_name = f'{file.split(".")[0]}_{lower_cut_point}.edgel'
            save_prompt_completion(prompt, completion, f'{output_folder}/{output_file_name}')

            prompt, completion = cut_graph(graph, middle_cut_point)
            output_file_name = f'{file.split(".")[0]}_{middle_cut_point}.edgel'
            save_prompt_completion(prompt, completion, f'{output_folder}/{output_file_name}')

            prompt, completion = cut_graph(graph, upper_cut_point)
            output_file_name = f'{file.split(".")[0]}_{upper_cut_point}.edgel'
            save_prompt_completion(prompt, completion, f'{output_folder}/{output_file_name}')

def cut_graph(serialized_graph: str, cut_point: int):
    '''
  Assumes the graph is given by a list of edges, separated by the new line symbol.
  Returns the first cut_point lines and all lines after cut_point lines until the last line.
  '''
    # separate by new line symbol
    lines = list(filter(None, serialized_graph.split('\n')))
    assert len(lines) >= cut_point

    # return prompt and completion
    return '\n'.join(lines[:cut_point]), '\n'.join(lines[cut_point:])

def reconstruct_graph(name, diff_graph_id, nodes=[], edges=[]):
    G = nx.DiGraph(name=name)
    G.diff_id=diff_graph_id
    for n in nodes:
        G.add_node(n[0], **n[1])
    for e in edges:
        if ((e[2]) != None):
            G.add_edge(e[0], e[1], **e[2])
        else:
            G.add_edge(e[0], e[1])

    return G
####################### END GRAPH MODIFICATIONS ########################