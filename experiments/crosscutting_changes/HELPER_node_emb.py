import ast
import json
import math
import os
import sys
import sys
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings



print("python path")
current_dir = os.path.dirname(os.path.abspath(__file__))
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFacePipeline
# Get the parent directory
parent_dir = os.path.dirname(current_dir)
# Get the grandparent directory (parent of the parent)
grandparent_dir = os.path.dirname(parent_dir)
# Add the parent directory to sys.path
sys.path.insert(0, parent_dir)

# Add the grandparent directory to sys.path
sys.path.insert(0, grandparent_dir)
print(sys.path)

CURRENT_LLM_MODEL ="text-embedding-3-small"

FALLBACK_LLM_MODEL = "all-MiniLM-L6-v2" #LLM_MODEL_EMBED = "text-embedding-3-small"
# Initialize the embeddings model and compute the embeddings
embedding_function_default = HuggingFaceEmbeddings(model_name=FALLBACK_LLM_MODEL)

# Initialize OpenAI client (ensure your API key is set up)
client = OpenAI(api_key="TODO: ADD YOURS")

def get_embedding_openai(text, model):
    return client.embeddings.create(input=[text], model=model).data[0].embedding

############################################################


def embed_documents_openai(documents, model):  # CHANGED: Now does batching
    batch_size = 1000  # NEW: batch size can be adjusted
    all_embeddings = []  # NEW
    for i in range(0, len(documents), batch_size):  # NEW
        batch_docs = documents[i : i + batch_size]  # NEW
        response = client.embeddings.create(input=batch_docs, model=model)  # CHANGED: call once per batch
        all_embeddings.extend([item.embedding for item in response.data])   # NEW
    return all_embeddings  # NEW

def get_node_embeddings(nodes, embedding_function= embedding_function_default):
    labels = [eval(data['label']) for node, data in nodes]
    labels_str = [str(label) for label in labels]
    if CURRENT_LLM_MODEL=="text-embedding-3-small":
        embeddings = embed_documents_openai(labels_str, CURRENT_LLM_MODEL)
    else:
        embeddings = embedding_function.embed_documents(labels_str)
    return dict(zip([node for node, data in nodes], embeddings))


def individual_embeddings(nodes, embedding_function= embedding_function_default): 
    attributes_list = [str(d['attributes']) for _, d in nodes]
    ids_list        = [str(d['node_id'])    for _, d in nodes]
    types_list      = [str(d['type'])       for _, d in nodes]
    class_list      = [str(d['class_name']) for _, d in nodes]

    if CURRENT_LLM_MODEL == "text-embedding-3-small":
        attr_embs  = embed_documents_openai(attributes_list, CURRENT_LLM_MODEL)
        id_embs    = embed_documents_openai(ids_list,        CURRENT_LLM_MODEL)
        type_embs  = embed_documents_openai(types_list,      CURRENT_LLM_MODEL)
        class_embs = embed_documents_openai(class_list,      CURRENT_LLM_MODEL)
    else:
        # Adjust if you use embed_documents vs embed_query
        attr_embs  = embedding_function.embed_documents(attributes_list)
        id_embs    = embedding_function.embed_documents(ids_list)
        type_embs  = embedding_function.embed_documents(types_list)
        class_embs = embedding_function.embed_documents(class_list)

    node_embedding_map = {}
    for i, (node, _) in enumerate(nodes):
        node_embedding_map[node] = {
            'attributes': attr_embs[i],
            'node_id':    id_embs[i],
            'type':       type_embs[i],
            'class_name': class_embs[i]
        }

    return node_embedding_map



def get_embeddings_complex(nodes, focus,embedding_function= embedding_function_default): 

    overall_focuses={}
    for node, data in nodes: 
        label_focus = eval(data['label']) 
        graph = data['graph']
        predecessors = list(data['graph'].predecessors(node))
        successors = list(data['graph'].successors(node))

        labels_predecessors ={}
        labels_successors = {}
        for p in predecessors: 
            try: 
                labels_predecessors[p]= ast.literal_eval(graph.nodes[p]['label'])
            except (ValueError, SyntaxError) as e:
                print(f"Error while computing focus evaluating node data for graph {graph.name}. Skipping node.")
                continue  # Skip the rest of the loop for this iteration
        for s in successors: 
            try: 
                labels_successors[s]= ast.literal_eval(graph.nodes[s]['label'])
            except (ValueError, SyntaxError) as e:
                print(f"Error while computing focus evaluating node data for graph {graph.name}. Skipping node.")
                continue  # Skip the rest of the loop for this iteration
          


        if (focus == "ALL"): 
        # Create result dictionary
            result = {
                'focus_node': label_focus,
                'successors':  labels_successors.values(),
                'predecessors': labels_predecessors.values()
            }

        if (focus == "SUC_PREC"): 
        # Create result dictionary
            result = {
                'successors':  labels_successors.values(),
                'predecessors': labels_predecessors.values()
            }
        
        if (focus == "SUC"): 
        # Create result dictionary
            result = {
                'successors':  labels_successors.values()
            }
        if (focus == "PREC"): 
        # Create result dictionary
            result = {
                'predecessors': labels_predecessors.values()
            }
        overall_focuses[node] = result
        
    embeddings = embedding_function.embed_documents([str(focus) for focus in overall_focuses.values()])
        
    return dict(zip([node for node, data in nodes], embeddings))






