import torch
import json
import networkx as nx
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

model_name = 'unsloth/Phi-4-mini-instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True
)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map='auto',
    torch_dtype=torch.float16,
    quantization_config=quantization_config
)

prompt = """Generate a JSON object representing a task execution DAG for the following ZebraLogic problem:
Solve ZebraLogic constraint grid: {houses: 5, attributes: [color, nationality, drink, pet, cigarette]}
The JSON must strictly follow this schema:
{"nodes": ["node1", "node2"], "edges": [{"source": "node1", "target": "node2", "relation": "depends_on"}]}
Return ONLY valid JSON.
JSON:"""

inputs = tokenizer(prompt, return_tensors='pt').to('cuda')
valid_count = 0
total = 20

print('Running baseline LLM evaluation...')
for i in range(total):
    outputs = base_model.generate(**inputs, max_new_tokens=400, do_sample=True, temperature=0.7)
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start != -1 and end != -1:
            data = json.loads(response[start:end])
            if 'nodes' in data and 'edges' in data:
                G = nx.DiGraph()
                G.add_nodes_from(data['nodes'])
                for edge in data['edges']:
                    G.add_edge(edge['source'], edge['target'])
                if nx.is_directed_acyclic_graph(G):
                    valid_count += 1
    except Exception as e:
        pass

print(f'Baseline Validity: {(valid_count / total) * 100}%')
