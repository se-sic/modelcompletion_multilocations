import networkx as nx
import re

def parse_puml(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    classes = {}
    relationships = []
    current_class = None

    for line in lines:
        class_match = re.match(r"class (\w+)(?: #(\w+))*", line)
        relationship_match = re.match(r'(\w+) "(\w+)" -- "(\w+)" (\w+) : (\w+) >', line)
        if class_match:
            current_class = class_match.group(1)
            color = class_match.group(2) if class_match.group(2) else 'grey'
            classes[current_class] = {"attributes": [], "methods": [], "color": color}
        elif relationship_match:
            relationships.append({
                "class1": relationship_match.group(1),
                "relationship1": relationship_match.group(2),
                "relationship2": relationship_match.group(3),
                "class2": relationship_match.group(4),
                "description": relationship_match.group(5)
            })
        elif current_class:
            attribute_match = re.match(r"\s*\+(?:<color:(\w+)>)*(\w+) : (\w+)(?:<\/color>)*", line) 
            method_match = re.match(r"\s*\+?(?:<color:\s*(\w+)>)\s*\+?(\w+)\((.*)\) : (\w+)\s*(?:<\/?color(:\w+)?>)", line)
           #  <color:green> +checkoutBook(book: Book, student: Student) : boolean <color:green>

            
            if attribute_match:

                color = attribute_match.group(1)  if attribute_match.group(1) else 'grey'
                attribute_name = attribute_match.group(2)
                classes[current_class]["attributes"].append({"name": attribute_name, "color": color})
            elif method_match:
                color = method_match.group(1) if method_match.group(1) else 'grey'
                method_name = method_match.group(2)
                method_params = method_match.group(3)
                return_type = method_match.group(4)
                classes[current_class]["methods"].append({"name": method_name, "params": method_params, "return": return_type, "color": color})
       
    return classes, relationships


def create_graph(classes, relationships):
    G = nx.DiGraph()
    for class_name, class_info in classes.items():
        G.add_node(class_name, label='class',  color=class_info['color'])
        for attribute in class_info["attributes"]:
            G.add_node(attribute["name"],label='attribute',  color=attribute['color'])
            G.add_edge(class_name, attribute["name"], label="class includes attribute")
        for method in class_info["methods"]:
            G.add_node(method["name"], label='method', color=method['color'])
            G.add_edge(class_name, method["name"], label="class includes method")
    
    for edge in relationships:
        G.add_edge(edge["class1"], edge["class2"], label=edge["description"])    


    return G



def generate_subgraphs(G, sorted_node_ids, radius):
    subgraphs = []
    for node_id in sorted_node_ids:
        subgraph = nx.ego_graph(G, node_id,undirected=True, radius=radius)
        subgraphs.append((node_id, subgraph))
    return subgraphs

def get_colored_subgraph(G):
    H = nx.DiGraph()
    focus_node=None
    for node, data in G.nodes(data=True):
        if data['color'] != 'grey':
            focus_node= node 
            H.add_node(node, label = data['label'], color=data['color'])
            for predecessor in G.predecessors(node):
                H.add_node(predecessor, **G.nodes[predecessor])
                H.add_edge(predecessor, node, **G[predecessor][node])
    return H, focus_node






