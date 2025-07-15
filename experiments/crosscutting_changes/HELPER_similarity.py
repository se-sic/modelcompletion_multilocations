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

from experiments.crosscutting_changes.HELPER_node_emb import FALLBACK_LLM_MODEL


def compute_similarity_between_nodes(changed_node_embedding, compare_to_list_embeddings):
    

    # Convert embeddings to matrices for cosine_similarity
    # The result is a 2D NumPy array (comparison_matrix) where each row represents an embedding vector.
    comparison_matrix = np.array(list(compare_to_list_embeddings.values()))
    if comparison_matrix.ndim == 1:
        comparison_matrix = comparison_matrix.reshape(1, -1)
    #comparison_matrix = np.array(list(compare_to_list_embeddings.values())).reshape(-1, len(compare_to_list_embeddings))

    # Purpose: Converts the changed_node_embedding into a 2D NumPy array to match the dimensions of comparison_matrix.
    # Details: Ensures the embedding is in the correct shape (1, n_features) for cosine similarity calculation
    changed_node_matrix = np.array(changed_node_embedding).reshape(1, -1)

    # Compute cosine similarity
    similarity_matrix = cosine_similarity(changed_node_matrix, comparison_matrix)

    # Calculate the average similarity
    average_similarity = np.mean(similarity_matrix)

    #return average_similarity
    return similarity_matrix, average_similarity


# Initialize the embeddings model and compute the embeddings
def compute_similarity_to_recent_change(recent_node, recent_focus_point, model_digraph):


    embedding_function = HuggingFaceEmbeddings(model_name= FALLBACK_LLM_MODEL)
    model_id = "sentence-transformers/all-MiniLM-L6-v2"

    def get_node_embeddings(G):
        embeddings = embedding_function.embed_documents(G.nodes())
        return dict(zip(G.nodes(), embeddings))

    # Get the embeddings
    node_embeddings = get_node_embeddings(model_digraph)
    subgraph_embeddings = get_node_embeddings(recent_focus_point)

    #TODO change this this is currently 
    changed_node = subgraph_embeddings[recent_node]

    # Convert the embeddings to a matrix for cosine_similarity
    G_matrix = np.array(list(node_embeddings.values()))
    change_node_matrix = np.array(list(changed_node))


    # Compute cosine similarity and sort nodes
    similarity_matrix = cosine_similarity(change_node_matrix.reshape(1,-1), G_matrix)
    most_similar_nodes_indices = np.argsort(-similarity_matrix)

    # Create a dictionary of most similar nodes
    node_ids = np.array(list(node_embeddings.keys()))
    sorted_node_ids = node_ids[most_similar_nodes_indices]
    print(sorted_node_ids)

    return sorted_node_ids
