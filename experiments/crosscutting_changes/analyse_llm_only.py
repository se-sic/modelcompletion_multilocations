import csv
import json
import os
import sys
import textwrap
import numpy as np

import networkx as nx




sys.path.insert(0, '/scratch/s8aawelt/GraphGeneration/')
sys.path.insert(0, 'GraphGeneration/')
cache_location = '/scratch/s8aawelt/tuning/.cache'
os.environ['HF_HOME'] = cache_location
os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_location
from experiments.crosscutting_changes.HELPER_node_emb import compute_similarity_to_recent_change
from experiments.crosscutting_changes.HELPER_construct_graph_from_uml import create_graph, generate_subgraphs, get_colored_subgraph, parse_puml
from modules.graph.io import _graph_to_dict

import math
import transformers
import torch
import textwrap
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

login("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")
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
   
)

device=torch.cuda.current_device()
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)


pipeline = transformers.pipeline(
        "text-generation", model=model,tokenizer=tokenizer,  model_kwargs={"torch_dtype": torch.bfloat16},
    device=torch.cuda.current_device()
)



instruction= f""" You are an expert at software modeling.

You are given two serialized snippets of a larger software model.
The first one named recent change, shows the part of the software model 
were recently a change was done to. If highlighted as green an element was added, 
if highlighted as red an element was removed. If highlighted as blue an element was changed.
The second snippet named current focus, shows the part of the software model that is currently being worked on. 
The current context includes all directly neighboring and connected items to the focus point, to better clarify how the current focus is embedded in the software model.
Your task a software modeler is to suggest further one specific, concrete change to the current context which is related and relevant to the recent change. 
Therefore copy and slightly adjust the current context and highlight in the corresponding change what you would like to change in the current context. 
Keep your responses brief and to the point, only include the modified current context according to the given format, nothing else. 
Provide the answer in the specified format according to the current content"""

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
def send_prompt_and_get_response(request_llm, max_prompt_length=math.inf):

    prompt = pipeline.tokenizer.apply_chat_template(
        request_llm,
        tokenize=False,
        add_generation_prompt=True
    )

    prompt_length = len(pipeline.tokenizer.encode(prompt))
    print("prompt length")
    print(prompt_length)




    if prompt_length > max_prompt_length:
        return prompt, "prompt with SCG is too long: " + str(prompt_length)

    result = pipeline(
        prompt,
        max_new_tokens=600,
        eos_token_id=terminators,
        do_sample=False,
        temperature=0.2,
        top_p =0.7,
        pad_token_id=pipeline.tokenizer.eos_token_id
    )

    formula = result[0]["generated_text"][len(prompt):]
    return prompt, formula







def build_request_cross_cutting(json_recent_change,current_focus, current_context):

    scenario = textwrap.dedent(instruction)

    #TODO i dont know what to best add here 
    return [
        {"role": "system", "content": scenario},
        {"role": "user", "content": "recent change focus: " + json_recent_change+"\n"+ "current focus: "+current_focus+"\n" + "current context: "+current_context+"\n"},
    ]





folder = 'data_simple/'

with open('output_crosscuttingresults.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['file_name', 'prompt_llm','recent_focus', 'current_focus', 'current_focus_prio', 'current_context' ,'result_llm'])

for filename in os.listdir(folder):
    #TODO make a loop around all of this data from the files 
    classes, relationships = parse_puml(folder+filename)

    model_digraph = create_graph(classes,relationships)

    recent_focus_point,recent_node_id = get_colored_subgraph(model_digraph)


    sorted_next_focus_points = compute_similarity_to_recent_change(recent_node_id, recent_focus_point, model_digraph)

    current_focus_subgraphs= generate_subgraphs(model_digraph, sorted_next_focus_points[0], 2)
    print("current_focus_subgraph")



    json_recent_change =json.dumps(_graph_to_dict(recent_focus_point, recent_focus_point.name), indent=4)
    json_current_focus =[( current_focus_subgraph[0], json.dumps(_graph_to_dict(current_focus_subgraph[1], current_focus_subgraph[1].name), indent=4))for current_focus_subgraph in current_focus_subgraphs ]




    #disgrad the very first item since it is the same as the recent change  
    # and only consider the next 3 items                
    json_current_focus = json_current_focus[1:4] 
    current_focus_prio=0
    for focus_candidate in json_current_focus:
       
        current_context = focus_candidate[1]
        current_focus_item = focus_candidate[0]
        
        llm_request = build_request_cross_cutting(json_recent_change,  current_focus_item, current_context)
        prompt, result = send_prompt_and_get_response(llm_request)
        with open('output_crosscuttingresults.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    row = [filename, llm_request , json_recent_change,  current_focus_item, current_focus_prio, current_context, result]
                    writer.writerow(row)
        current_focus_prio+=1

    #TODO check if results goes in the right direction of the ground truth


