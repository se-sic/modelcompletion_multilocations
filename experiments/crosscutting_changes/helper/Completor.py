from abc import ABC
import json
import os
import random
import sys
import networkx as nx
from networkx.readwrite import json_graph
import ast
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)

sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)
from experiments.crosscutting_changes.helper.vectordb_Tinnes import ChangeGraphVectorDB


from experiments.crosscutting_changes.helper.HELPER_GENERIC import _graph_to_dict_general, save_to_json
from experiments.crosscutting_changes.helper.HELPER_final_eval import _prune_row
from experiments.crosscutting_changes.helper.HELPER_node_lables import compute_change_class, remove_certain_info

from experiments.crosscutting_changes.helper.HELPER_connectLLM import OpenAILLM, load_openai_api_key
from experiments.crosscutting_changes.helper.baseline_methods_Tinnes import CHAT_MODEL_INSTRUCTION_TINNES, CHAT_MODEL_INSTRUCTION_OURS, CHAT_MODEL_INSTRUCTION_TINNES, CHAT_MODEL_INSTRUCTION_TINNES_ADJUSTED, PREDICTIONS_BASE_INSTRUCTION_OURS, V_JSON, CosineSimilarity, calculate_change_type_correctness, calculate_structure_correctness, calculate_type_correctness, get_component_of_node, parse_graph, serialize_graph, transform_gen_graph, transform_gen_list, transform_gt_entry
N = 10 
file_gT_nextFocus = "groundTruth_nextFocus"
file_gT_localCompletion = "groundTruth_localCompletion"
#'{\n  "nodes": [],\n  "edges": [\n    {\n      "source": "4",\n      "target": "369",\n      "label": "{\'changeType\': \'Preserve\', \'type\': \'reference\', \'referenceTypeName\': \'ePackage\'}"\n    }\n  ]\n}'

#DEFAULT_EXAMPLE= '{\n  "nodes": [\n    {\n      "id": "999",\n      "label": "{\'type\': \'object\', \'className\': \'EAnnotation\', \'attributes\': {\'id\': \'_NewEAnnotationId\', \'source\': \'http://www.eclipse.org/emf/2002/GenModel\', \'details\': \'{}\'}}"\n    }\n  ],\n  "edges": [\n    {\n      "source": "4",\n      "target": "999",\n      "label": "{\'changeType\': \'Preserve\', \'type\': \'reference\', \'referenceTypeName\': \'eAnnotations\'}"\n    }\n  ]\n}'
DEFAULT_EXAMPLE="t #  diff_8.json \ne 29 31 {'changeType': 'Change', 'type': 'attribute'} {'changeType': 'Preserve', 'type': 'object', 'className': 'EAttribute', 'attributes': {'id': 'QVTOperational.ModuleImport.kind', 'name':'kind','ordered':'true','unique':'true','lowerBound':'0','upperBound':'1','many':'false','required':'false','eType':'ImportKind','eGenericType':'org.eclipse.emf.ecore.impl.EGenericTypeImpl@18724279 (expression: ImportKind)','changeable':'true','volatile':'false','transient':'false','defaultValue':'extension','unsettable':'false','derived':'false','eContainingClass':'ModuleImport','iD':'false','eAttributeType':'ImportKind'}} {'changeType': 'Change', 'type': 'attributeValue', 'className': 'EAttribute', 'attributeName': 'defaultValueLiteral', 'valueBefore': 'access', 'valueAfter': 'null', 'attributes': {'id': 'QVTOperational.ModuleImport.kind_11'}}"
DEFAULT_EXAMPLE="t # diff_8.json\ne 29 32 {'changeType': 'Change', 'type': 'attribute'} _ {'changeType': 'Change', 'type': 'attributeValue', 'className': 'EAttribute', 'attributeName': 'ordered', 'valueBefore': 'true', 'valueAfter': 'false'}"
DEFAULT_EXAMPLE='t # diff_8.json\ne 29 30 "{\'changeType\': \'Change\', \'type\': \'attribute\'}" _ "{\'changeType\': \'Change\', \'type\': \'attributeValue\', \'className\': \'EAttribute\', \'attributeName\': \'ordered\', \'valueBefore\': \'true\', \'valueAfter\': \'false\'}"'
models_dir =""
include_rag = True
query_LLM = True
radius_important = True
CHEAP = True
star_ramc=True
baseline = "JSON" #"edgL" #"JSON" # "JSON" if edgl, slicing etc. everything like tinnes et al, also LLM
# if Json, other scliing, other propmpt etc.

class Completor(ABC):

    instruction= CHAT_MODEL_INSTRUCTION_TINNES_ADJUSTED if baseline== "JSON" else CHAT_MODEL_INSTRUCTION_TINNES

    

    def __init__(self, path_results, path_db, whatToRemove):
        self.correctness_focus = []
        self.path_results = path_results
        self.path_db = path_db
        self.vector_db = ChangeGraphVectorDB(self.path_db)
        self.vector_db.load_existing()
        self.whatToRemove = whatToRemove

    def complete_model(self, graph, item_starting_node, max_completions, test_file_map, out_final_eval ):
        print("Completing model with next focus")
        already_considered_nodes = [item_starting_node]

        num_iterations = N 
        if max_completions < num_iterations: 
            num_iterations = max_completions-1 #-1 becuase we randomly pick one at beginning and than directly antoher on 
        for n in range(num_iterations):  # assuming N is defined elsewhere
            next_focus_node = self.next_focus(item_starting_node, out_final_eval, already_considered_nodes, graph)
            # if wrong we wont consider this focus node anymore 
            if next_focus_node<0: 
                break

            slice_ = self.computeSlice(graph, next_focus_node)

            result = self.local_completion_LLM(self.instruction, slice_, n, next_focus_node, graph)

            # resets also depending on the metric, focus_node and model_completion stuff 
            # updates also the graph, the list, and if 
            graph, next_starting_node, results_metrics = self.evaluate_output(graph, result, next_focus_node, n)

            save_to_json(self.path_results, f"results_metrics{n}", results_metrics)
           
            #TODO outdate all files for the following iteration, so ground truth 
            # update the graph but also make sure embedding, changed etc is removed
            already_considered_nodes.append(next_focus_node)
            item_starting_node= int(next_starting_node)

    def add_nodes_edges(self, graph,source_node, ground_truth_to_next_focus, round):
        next_starting_nodes=[]
        for entry in ground_truth_to_next_focus:
            node_id = entry["successor"]
            next_starting_nodes.append(node_id)
            node_data = entry.get("node_data", {})
            edges = entry.get("edges", [])

            # --- Add node ---
            if node_id not in graph.nodes:
                graph.add_node(node_id)
                graph.nodes[node_id].update(node_data)
                print(f"[WARN] Skipping edge: target node {node_id} not in graph.")

            #only if random selcetion occured
            if source_node not in graph.nodes:
                gt_path = os.path.join(self.path_results, f"groundTruth_nextFocus{round}.json")
                with open(gt_path, "r") as f:
                    ground_truth_nextFocus = json.load(f)

                graph.add_node(source_node)
                graph.nodes[source_node].update(ground_truth_nextFocus[source_node])
                  
                print(f"[WARN] Skipping edge: source node {source_node} not in graph.")

            # --- Add edges ---
            for edge_info in edges:
               
                # Each edge must have a 'label' field like "{'changeType': 'DELETE', ...}"
                label_raw = edge_info.get("label", "{}")
                
                label_dict = {"label": label_raw}
                
                # For now, connect from existing node(s) → this node
                # You can adapt this direction logic as needed
                
                graph.add_edge(source_node, node_id, **label_dict)

        if not next_starting_nodes: 
            next_starting_nodes = [random.choice(list(graph.nodes())) ]    

        return graph, next_starting_nodes

    


    def local_completion_LLM(self, instruction, slice_, n, next_focus_node, graph):

        print("Completing model all at once")
       

        # remove all info the LLM does not have to guess correctly 
        # deside whether rag or not
        if (include_rag):
            if (baseline=="JSON"): 
                loaded = ast.literal_eval(slice_)
                H = json_graph.node_link_graph(loaded,directed=True)
                json_slice = slice_
                   
                H.diff_id = graph.diff_id

                slice_ = serialize_graph(H, version=V_JSON)[0]

            examples = self.get_few_shot_examples(slice_, nb_few_shot_samples=3)
            
            if (baseline=="JSON"):
                examples_json =[]
                for g in examples: 
                    _, few_shots_to_graph = self.transform_LLM_output(graph, g.page_content, skipJson=True)
                    few_shots_to_json = json.dumps( _graph_to_dict_general(few_shots_to_graph, few_shots_to_graph.name), ensure_ascii=False)
                    examples_json.append(few_shots_to_json)

                examples=examples_json
                slice_=json_slice
                examples_str = "\n---\n".join(str(ex) for ex in examples)
            else:

                examples_str = examples_str = "\n---\n".join(ex.page_content for ex in examples)
            
            prompt_str = (
                instruction.strip() + "\n\n"
                + examples_str + "\n---\n"
                + slice_.strip()
            )
            if baseline=="JSON": 
                prompt_str +=  f" The source node of the edge is already given by: {str(next_focus_node)}, what additional node should be added as an successor of this node."
            
        else: 
            prompt_str= instruction + slice_ + f" The source node of the edge is already given by: {str(next_focus_node)}, please add the ONE additional node that is a successor of this node and their corresponding connecting edge."

        # GPT-4 (version 0613) 'gpt-4-0613' -> very expensive Why they’re still around Backward compatibility (old scripts depending on exact model name)
        #text‑embedding‑3‑small cheapest gpt-5-nano
        api_key = load_openai_api_key()
        if baseline=="JSON": 
            llm = OpenAILLM(model_id='gpt-5-mini', api_key=api_key)
            message=[{"role": "user", "content": prompt_str}]
        # print (llm.list_models())
        # this is for text completion models
        else: 
            if CHEAP: 
                llm = OpenAILLM(model_id='gpt-5-mini', api_key=api_key)
                message = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt_str}]
    
            else: 
                llm = OpenAILLM(model_id='gpt-4-0613', api_key=api_key)
                message = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt_str}]
        
        if (query_LLM): 
            if baseline=="JSON": 
                result = llm.query_chat(message)
            else: 
                if CHEAP: 
                    result = llm.query_chat(message)
                else: 
                    result = llm.query_text(message)
        else: 
            result =DEFAULT_EXAMPLE

        os.makedirs(self.path_results, exist_ok=True)
        output_path = os.path.join(self.path_results, f"iteration_{n}_model_completion.txt")
        
        with open(output_path, "w") as output_file:
            output_file.write(result)

        prompt_path = os.path.join(self.path_results, f"iteration_{n}_prompt.txt")
        with open(prompt_path, "w") as prompt_path:
            prompt_path.write(prompt_str)
        
        return result

    def evaluate_next_Focus(self, next_focus_node, round, graph): 
        #TODO check whether the new focus node is in path_results/
        # was saved like this save_to_json(path_results,"groundTruth_nextFocus", groundTruth_nextFocus )
        # which includes list of  isPredessor_nodes = []
        gt_path = os.path.join(self.path_results, f"groundTruth_nextFocus{round}.json")
        with open(gt_path, "r") as f:
            ground_truth = json.load(f)

        correct_next_focus = str(next_focus_node) in ground_truth.keys()

        # if false we take out a random item from ground
        if not correct_next_focus: 
            random.seed(42)
            changed, notchanged = compute_change_class(graph, False, False)
            #next_focus_node = int(random.choice(list(ground_truth.keys())))
            if changed:
                next_focus_node = int(random.choice(changed)[0])

            else:
                next_focus_node = int(random.choice(graph.nodes()))
            # add the node to the graph 
            #if next_focus_node not in graph.nodes:
            #    graph.add_node(str(next_focus_node))
            #    graph.nodes[str(next_focus_node)].update(ground_truth[str(next_focus_node)])
        
        # TODO and the next_focus_node is remove from the ground_truth_list and saved under
        gt_file_new =  f"groundTruth_nextFocus{round+1}"
        ground_truth.pop(str(next_focus_node), None)
        save_to_json(self.path_results, gt_file_new, ground_truth)
    
        return correct_next_focus, next_focus_node
    
    def evaluate_LLM_output(self,graph, result, next_focus_node, round):

        result_summary = {
            "format_correct": False,
            "structure_correct": False, 
            "change_correct": False,
            "type_structure": False,
            "semantic_correctness_transfered": False
        }

        try:
            isparsable, result_loaded = self.transform_LLM_output(graph, result)
            result_summary["format_correct"] = isparsable                

        except Exception as e:
            isparsable = False
            print("format already wrong")
        

        
        gt_path = os.path.join(self.path_results, f"groundTruth_localCompletion{round}.json")
        with open(gt_path, "r") as f:
            ground_truth = json.load(f)
            if str(next_focus_node) in ground_truth:
                ground_truth_to_next_focus = ground_truth[str(next_focus_node)]
                if isparsable: 
                    # TODO: add further checks here
                    # e.g. structure, change, type_structure, etc.
                    transformed_gT  = [transform_gt_entry(e, source_node=str(next_focus_node)) for e in ground_truth_to_next_focus]
                    transformed_gen = self.transform_gen_graph_approach(result_loaded, source_node=str(next_focus_node))
                    #TODO transofrm to comparable format 
                    result_summary["structure_correct"] = calculate_structure_correctness(next_focus_node, transformed_gT, transformed_gen)
                    #if result_summary["structure_correct"]: 
                    result_summary["change_correct"] = calculate_change_type_correctness(transformed_gT, transformed_gen )
                    result_summary["type_structure"] = calculate_type_correctness(transformed_gT , transformed_gen)
                        #result_summary["semantic_correctness_transfered"] = CosineSimilarity().compute(result_loaded,transformed_gT )

            else:
                # could be the case if the suggested node by LLM is not in graph, 
                # or there is simply no next focus node left 
                # in this case we add some random from the gT to the model to make it possible again to be crorrect again 
                #next_focus_node, ground_truth_to_next_focus = random.choice(list(ground_truth.items()))
                # print("all wrong sampling new node")
                # we are not doing this anymore since it would lead to the fact that we slighlty would prefer 
                # the method performing worse, since it gets new chances all the time 
                # TODO but else it is always just fucking bad , the method perfromance
                if ground_truth:
                    next_focus_node, ground_truth_to_next_focus = random.choice(list(ground_truth.items()))
                else:
                    next_focus_node, ground_truth_to_next_focus = random.choice(list(graph.nodes())), []      


        # now we fix the graph, the ground truth for the iteration
        # save_to_json(path_results_filename, "existingGraph0", _graph_to_dict(existing_graph), existing_graph.diff_id)
        graph, next_starting_nodes = self.add_nodes_edges(graph,str(next_focus_node), ground_truth_to_next_focus , round)
        save_to_json(self.path_results, f"existingGraph{round+1}", _graph_to_dict_general(graph, graph.diff_id))


        ground_truth.pop(str(next_focus_node), None) #, is not neccesary to be removed, alwasy the same
        #if node is more the same so it is
        save_to_json(self.path_results, f"groundTruth_localCompletion{round+1}", ground_truth)

        #TODO we need to check if[0] is ok
        return graph,next_starting_nodes[0], result_summary
       


    def computeSlice(self,  graph, next_focus_node): 

        slice_ = get_component_of_node(graph, next_focus_node)
        #required because get_component_of_node is partically adding them again 
        slice_ = remove_certain_info(slice_, self.whatToRemove)
        serialized_graph = serialize_graph(slice_, version=V_JSON)[0]
        return serialized_graph      

    def transform_gen_graph_approach(self, result_loaded, source_node):
        return transform_gen_graph(result_loaded, source_node)
    
    def transform_LLM_output(self, graph, result, skipJson=False): 

        worked, result =  parse_graph(result)
        empty_nodes = [n for n, d in result.nodes(data=True) if d.get('label') in ('', '"{}"', '{}')]
        for empty_node in empty_nodes:
            if str(empty_node) in graph.nodes:
                result.nodes[empty_node].update(graph.nodes[str(empty_node)])
        return worked, result
    

    def get_few_shot_examples(self, test_sample, nb_few_shot_samples): 
       
        # as done in the original paper 
        selected_few_shot_samples = self.vector_db.query_k_most_diverse_strong(test_sample,
                                                                              scope="train", k=nb_few_shot_samples,
                                                                              k_retrieve=nb_few_shot_samples * 2,
                                                                              num_of_iterations=10)
    
        return selected_few_shot_samples
    

    def parse_LLM_output(self,graph, result): 
        result_parsable = False
        try:
            worked, result_loaded = self.transform_LLM_output(graph, result)
            result_parsable = worked                

        except Exception as e:
            print("format already wrong")
            result_parsable = False   
            result_loaded= None
        
        return result_parsable, result_loaded
           

class AllAtOnceCompletor(Completor):
 
    isNextFocus= False

    def __init__(self, path_results, path_db, whatToRemove):
        super().__init__(path_results, path_db, whatToRemove)
        self.correctness_focus = []
        self.path_results = path_results
        self.path_db = path_db
        self.whatToRemove = whatToRemove


    #DONE according to Tinnes et al. the location where the completion is done is always the same 
    #only difference is, 
    def next_focus(self, item_starting_node, out_final_eval, already_considered_nodes, graph):
        return item_starting_node
    
    def evaluate_output(self, graph, result, next_focus_node, round): 

        # TODO we first have to eval the output from the LLM, otherwise for Tinnes et al. 
        # we cant check wether nextFocus is correct or not 
        # result here is a string vom an edge 
    
        isParseable, output = self.parse_LLM_output(graph, result)
        # we cant check wether nextFocus is correct or not 
        # result here is a string from an edge 

        if isParseable:
            # get the next focus from this, the first 0 ist for first edge, second 0 for the source node 
            next_focus_node = list(output.edges(data=True))[0][0] #all empty
        # the source node is definitly wrong, however the descision how to proceed is made in evaluate_next_Focus
        else: 
            next_focus_node = -1 

        correct_next_focus,  next_focus_node = self.evaluate_next_Focus(next_focus_node, round, graph)

       
     
        graph, next_starting_node, correctness_levels = self.evaluate_LLM_output(graph, result, next_focus_node, round)
        next_starting_node = next_focus_node
        correctness_levels["correct_next_focus"] = correct_next_focus

        return graph, next_starting_node, correctness_levels
    
        



class NextFocusCompletor(Completor):
    
    isNextFocus= True

    def __init__(self, path_results, path_db,  whatToRemove):
        super().__init__(path_results, path_db, whatToRemove)
        self.instruction = (
            PREDICTIONS_BASE_INSTRUCTION_OURS
            if baseline == "JSON"
            else super().instruction
        )
        self.correctness_focus = []
        self.path_results = path_results
        self.path_db = path_db
        self.whatToRemove = whatToRemove
   
    def transform_gen_graph_approach(self, result_loaded, source_node):
        #if baseline=="JSON": 
         #   return transform_gen_list(result_loaded, source_node)
        #else: 
        return super().transform_gen_graph_approach( result_loaded, source_node)

    #TODO in here the filterting linearized_graph_data for node which only exist need to be done
    # die ersten node ids, also müssen wir das danach auch filtern , alle raus die nicht im graph

    def evaluate_output(self, graph, result, next_focus_node, round): 

        # we cant check wether nextFocus is correct or not 
        # result here is a string from an edge 
        correct_next_focus,  next_focus_node = self.evaluate_next_Focus(next_focus_node, round, graph)

        # this is for an edge case scenario, were actual focus node cannot be represented by edgL format
        # this is a huge drawback of the format since it cannot represent node if they are standing alone 
        isParseable, output = self.parse_LLM_output( graph, result)
        if isParseable:
            # wenn das next focus flasch war dann ist next focus node schon mal gut anhaltspunkt
            if not correct_next_focus and baseline=="JSON":
                #to nothin 
                v=0
            else: 
                # get the next focus from this, the first 0 ist for first edge, second 0 for the source node 
                next_focus_node_LLM = list(output.edges(data=True))[0][0] #all empty
                if int(next_focus_node_LLM) != next_focus_node: 
                    next_focus_node = int(next_focus_node_LLM)

     
        graph, next_starting_node, correctness_levels = self.evaluate_LLM_output(graph, result, next_focus_node, round)
     
        correctness_levels["correct_next_focus"] = correct_next_focus

        return graph, next_starting_node, correctness_levels
    
    def transform_LLM_output(self, graph, result, skipJson=False): 
        if baseline=="JSON" and not skipJson:
            try:
                loaded =  json.loads(result) 
                if "edges" in loaded:
                    loaded["links"] = loaded.pop("edges")
                    H = json_graph.node_link_graph(loaded,directed=True)
                    return True,  H
                else: 
                    return False, None
            except Exception:
                return False, None
        else: 
            return super().transform_LLM_output(graph, result)
    
                       
    def next_focus(self, item_starting_node, out_final_eval, not_to_consider, graph ):
       
        metas_f  = out_final_eval["meta"]
        probs_f  = out_final_eval["probability"]
        labels_f = out_final_eval["label"]

        metas_list = metas_f.iloc[0]   # [[102, 2, 4, 2], [102, 2, 9, 4], ...]
        probs_list = probs_f.iloc[0]   # [0.7567, 0.7151, 0.6530, ...]

        # now filter by the second element of each meta, needs to be same as current focis node
        filtered = [
            (i, meta, prob)
            for i, (meta, prob) in enumerate(zip(metas_list, probs_list))
            if meta[1] == item_starting_node
        ]
        # x[2] → the probability, reverse=True → highest first
        filtered = sorted(filtered, key=lambda x: x[2], reverse=True)

        if radius_important: 
            # 2) Keep top 10
            filtered = filtered[:10]
            # 3) Resort those top 10 by the 4th element of meta (meta[3])
            filtered = sorted(filtered, key=lambda x: x[1][3])

        #not_to_consider_nodes_of_starting_node = not_to_consider.get(item_starting_node, [])
        next_valid = next(
            (meta for (_, meta, _) in filtered if meta[2] not in not_to_consider and str(meta[2]) in graph.nodes()),
            (None, None, -1, None)  # fallback if all are excluded
        )
        print("Selecting next focus node")
        return next_valid[2] #the next node 
    
    #DONE computes the slice and returns its string representation
    def computeSlice(self, graph, next_focus_node): 
            
        if baseline =="JSON": 
            subgraph = nx.ego_graph(graph, str(next_focus_node), radius=2)
            subgraph.diff_id = graph.diff_id 
            # this info does absolutly not to be allowed available here 
            #subgraph = remove_certain_info(subgraph, self.whatToRemove)
            return str(_graph_to_dict_general(subgraph, subgraph.diff_id))
        else: 
            return super().computeSlice(graph, next_focus_node)




    