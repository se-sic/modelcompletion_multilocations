#!/usr/bin/env python3
import csv
import json
import os
import sys
import textwrap
import numpy as np
import networkx as nx

#sys.path.insert(0, '/scratch/s8aawelt/GraphGeneration/')
# sys.path.insert(0, 'GraphGeneration/')
##cache_location = '/scratch/s8aawelt/tuning/.cache'
#os.environ['HF_HOME'] = cache_location
#os.environ['SENTENCE_TRANSFORMERS_HOME'] = cache_location

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_dir, '..')))
sys.path.append(os.path.abspath(os.path.join(script_dir, '../..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
print("sys.path:", sys.path)
sys.path.insert(0, 'GraphGeneration/')

from experiments.crosscutting_changes.compute_embeddings import compute_similarity_to_recent_change
from experiments.crosscutting_changes.construct_graph import create_graph, generate_subgraphs, get_colored_subgraph, parse_puml
from modules.graph.io import _graph_to_dict
from torch.nn import DataParallel
from accelerate import Accelerator
import math
import transformers
import torch
import textwrap
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer

login("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")
model_id = "meta-llama/Meta-Llama-3-70B-Instruct"
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


if torch.cuda.is_available():
    current_device = torch.cuda.current_device()
    print("Available GPUs:", torch.cuda.device_count())
   # print(f"Current CUDA device: {current_device}")
   # print(f"Device name: {torch.cuda.get_device_name(current_device)}")
else:
    print("CUDA is not available.")
# Unset CUDA_VISIBLE_DEVICES
#if "CUDA_VISIBLE_DEVICES" in os.environ:
 #   del os.environ["CUDA_VISIBLE_DEVICES"]

def check_cuda():
    print("torch.cuda.is_available:", torch.cuda.is_available())
    print("torch.cuda.device_count:", torch.cuda.device_count())

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available.")

def check_environment_variables():
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "Not Set")
    print("CUDA_VISIBLE_DEVICES:", cuda_visible_devices)

def check_cuda_installation():
    print("Checking CUDA installation...")
    os.system("nvcc --version")
    os.system("nvidia-smi")

print("Checking CUDA setup...")
check_environment_variables()
check_cuda_installation()
check_cuda()

# Ensure the paths are resolved relative to the script's location
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
# Construct the absolute path for the 'data_simple/' directory
data_simple_path = os.path.join(parent_dir, 'data_simple')
accelerator = Accelerator()
print("Accelerator State:")
print("Number of processes:", accelerator.num_processes)
print("Process index:", accelerator.process_index)
#print("Is distributed:", accelerator.is_distributed)
device = accelerator.device
print(device)
#torch.cuda.empty_cache()
#torch.cuda.memory_summary(device=None, abbreviated=False)
accelerator.wait_for_everyone()
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="sequential",
    torch_dtype=torch.float16)

# Meta-Llama-3-70B-Instruct.Q2_K.gguf
#parent_dir = os.path.dirname(os.path.dirname(script_dir))
#model_id = os.path.join(parent_dir,"Meta-Llama-3-70B-Instruct.Q2_K.gguf")
#model_path = "Meta-Llama-3-70B-Instruct.Q4_K_S.gguf"
#model_path=  os.path.join(script_dir,"Meta-Llama-3-70B-Instruct.Q4_K_S.gguf")
#model_id = "MaziyarPanahi/Meta-Llama-3-70B-Instruct-GGUF"
#model_path= "Meta-Llama-3-70B-Instruct.Q2_K.gguf"
#model = AutoModelForCausalLM.from_pretrained(
 # model_id, gguf_file=model_path, 
  #  device_map={"": "cuda" if torch.cuda.is_available() else "cpu"},
   # torch_dtype=torch.bfloat16
#)
#if torch.cuda.device_count() > 1:
#    print(f"Using {torch.cuda.device_count()} GPUs!")
#    model = torch.nn.DataParallel(model)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id)
#tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=model_path)
#sys.exit(0)
model = accelerator.prepare(model)


tokenizer = accelerator.prepare(tokenizer)
if False:  # Replace with your condition
    print("Exiting the script")
    sys.exit(0)
pipeline = transformers.pipeline(
        "text-generation", model=model,tokenizer=tokenizer,  model_kwargs={"torch_dtype": torch.float16})

#pipeline = transformers.pipeline(
#        "text-generation", model=model,tokenizer=tokenizer,  model_kwargs={"torch_dtype": torch.float16},
#    device= device
#)

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
    with torch.no_grad():
        result = pipeline(
            prompt,
            max_new_tokens=600,
            eos_token_id=terminators,
            do_sample=False,
            temperature=0.2,
            top_p =0.7,
            pad_token_id=pipeline.tokenizer.eos_token_id
        )
    torch.cuda.empty_cache()
    formula = result[0]["generated_text"][len(prompt):]
    return prompt, formula


def build_request_cross_cutting(json_recent_change,current_focus, current_context):

    scenario = textwrap.dedent(instruction)

    #TODO i dont know what to best add here 
    return [
        {"role": "system", "content": scenario},
        {"role": "user", "content": "recent change focus: " + json_recent_change+"\n"+ "current focus: "+current_focus+"\n" + "current context: "+current_context+"\n"},
    ]
with open('output_crosscuttingresults.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    # Write the header
    writer.writerow(['file_name', 'prompt_llm','recent_focus', 'current_focus', 'current_focus_prio', 'current_context' ,'result_llm'])

for filename in os.listdir(data_simple_path):
    #TODO make a loop around all of this data from the files 
    classes, relationships = parse_puml(os.path.join(data_simple_path, filename))

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

        print(result)
        with open('testing_output_crosscuttingresults.csv', 'a', newline='') as file:
                    writer = csv.writer(file)
                    row = [filename, llm_request , json_recent_change,  current_focus_item, current_focus_prio, current_context, result]
                    writer.writerow(row)
        current_focus_prio+=1
        if True:  # Replace with your condition
            print("Exiting the script")
            sys.exit(0)
    break

 #TODO check if results goes in the right direction of the ground truth


