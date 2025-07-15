import ast
import os
import sys




current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Get the grandparent directory (parent of the parent)
grandparent_dir = os.path.dirname(parent_dir)

sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)

from modules.domain_specific.change_graph import clean_up_string_ast_literal






def compute_change_class(graph, withEmbeddings=True):

    skipped_nodes = 0
    preserve_nodes = []
    deleted_nodes = []
    added_nodes = []
    changed_nodes = []
    

    # Iterate through the nodes and categorize them
    for node, data in graph.nodes(data=True):
        # Extract the 'changeType' value from the 'label' attribute
        try: #node_data = eval(data['label'])
            # node_data = json.loads(data['label'])
            node_data = ast.literal_eval(clean_up_string_ast_literal(data['label']))
            node_id=node_data['attributes']['id']
            if withEmbeddings:
                embedding = ast.literal_eval(data['embedding'])
            isPredessor = ast.literal_eval(data['isPredessor'])  
            hasChangedNeightbor =  ast.literal_eval(data['isPredessor']) 


        except (ValueError, SyntaxError) as e:
            print(f"Error evaluating node data for graph {graph.name}. Skipping node.")
            skipped_nodes+=1
            continue  # Skip the rest of the loop for this iteration


        change_type = node_data.get('changeType', None)
        #print("change_type")
        #print(change_type)

        # Categorize the nodes based on the 'changeType'
        if withEmbeddings: 
            tuple_info = (node, {'label': str(node_data) , 'embedding': embedding ,'node_id': node_id, 'isPredessor':isPredessor, 'hasChangedNeighbor': hasChangedNeightbor, 'graph': graph }   )
        else: 
            tuple_info = (node, {'label': str(node_data) , 'node_id': node_id,'isPredessor':isPredessor,  'hasChangedNeighbor': hasChangedNeightbor,'graph': graph }   )

        if change_type == 'Preserve':
            preserve_nodes.append(tuple_info)
        elif change_type == 'Deleted' or change_type == 'Remove' or change_type == 'Delete':
            deleted_nodes.append(tuple_info)
        elif change_type == 'Change':
            changed_nodes.append(tuple_info)
        elif change_type == 'Add':
            added_nodes.append(tuple_info)
        else:
            # Raise an error if the changeType is none of the expected values
            raise ValueError(f"Unexpected changeType: {change_type} for node {node}")

    merged_changed_nodes = deleted_nodes + added_nodes + changed_nodes
    return merged_changed_nodes, preserve_nodes


def mark_neightbors(graph):
    skipped_nodes = 0
    flagged_as_predecssor_of_changed = set()
    flagged_as_successor_of_changed = set() 
    all_nodes=[]

    # we need to first find all the preserved stuff
    for node, data in graph.nodes(data=True):
        try: #node_data = eval(data['label'])
            # node_data = json.loads(data['label'])
            node_data = ast.literal_eval(clean_up_string_ast_literal(data['label']))

        except (ValueError, SyntaxError) as e:
            print(f"Error evaluating node data for graph {graph.name}. Skipping node.")
            skipped_nodes+=1
            continue

        change_type = node_data.pop('changeType', None)

        if change_type != 'Preserve':
                predessors = list(graph.predecessors(node))
                successors = list(graph.successors(node))

                for p in predessors:
                    flagged_as_predecssor_of_changed.add(p)

                for s in successors: 
                    flagged_as_successor_of_changed.add(p)
                    

        #please only add if really everything worked
        all_nodes.append(node)

    # add the embeddings to the graph
    for node in all_nodes:

        graph.nodes[node]["isPredessor"]= str(False)
        graph.nodes[node]["isSuccessor"]= str(False)

    for n in flagged_as_predecssor_of_changed:
        graph.nodes[n]["isPredessor"] = str(True)

    for s in flagged_as_successor_of_changed: 
        graph.nodes[s]["isSuccessor"] = str(True)


    return graph
