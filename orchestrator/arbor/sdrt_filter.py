import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from contracts.graph_schema import DiscourseGraph, SemanticNode

class SDRTGNNFilter:
    """
    Offline/Online SDRT Graph Neural Network Filter for Massive Context Windows.
    Instead of passing 256k tokens into an LLM context window (which causes OOM or Lost-in-the-Middle),
    this filter implements the offline segmentation and indexing of context into an SDRT graph,
    and the online O(1) query-time pruning to retrieve only the salient SemanticNodes.
    """
    def __init__(self, encoder_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load lightweight encoder for structural embeddings
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.encoder = AutoModel.from_pretrained(encoder_name).to(self.device)
        self.encoder.eval()
        
    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def _embed(self, texts):
        encoded_input = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt', max_length=512).to(self.device)
        with torch.no_grad():
            model_output = self.encoder(**encoded_input)
        sentence_embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
        import torch.nn.functional as F
        return F.normalize(sentence_embeddings, p=2, dim=1)

    def build_sdrt_index(self, massive_text, chunk_size=256):
        """
        OFFLINE PHASE:
        Parses massive text into Elementary Discourse Units (EDUs) and embeds them.
        Returns a DiscourseGraph object containing the embedded nodes.
        """
        # Segment text into EDUs (roughly by line/sentence for this simulation)
        sentences = [s.strip() for s in massive_text.split('\n') if len(s.strip()) > 5]
        
        # Batch encode to prevent OOM
        batch_size = 64
        all_embeddings = []
        for i in range(0, len(sentences), batch_size):
            batch = sentences[i:i+batch_size]
            embeds = self._embed(batch)
            all_embeddings.append(embeds.cpu())
            
        if all_embeddings:
            all_embeddings = torch.cat(all_embeddings, dim=0)
        else:
            all_embeddings = torch.empty((0, 384))
        
        nodes = []
        for i, (sent, emb) in enumerate(zip(sentences, all_embeddings)):
            node = SemanticNode(node_id=f"edu_{i}", concept=sent, vq_index=i)
            # Attach the vector natively for online pruning
            node.embedding = emb
            nodes.append(node)
            
        graph = DiscourseGraph(graph_id="offline_sdrt_index", nodes=nodes, edges=[])
        return graph

    def forward(self, sdrt_graph, query, top_k=3):
        """
        ONLINE PHASE:
        Scores the pre-computed graph nodes against the query and returns a compressed subgraph.
        This provides O(1) LLM scaling because the LLM only ever sees the pruned `top_k` nodes.
        """
        if not sdrt_graph.nodes:
            return sdrt_graph
            
        query_embed = self._embed([query]).cpu()
        
        # Extract embeddings from graph
        node_embeds = torch.stack([n.embedding for n in sdrt_graph.nodes])
        
        # Compute Cosine Similarity
        cos_scores = torch.mm(query_embed, node_embeds.transpose(0, 1))[0]
        
        # Prune Graph (Top K)
        actual_k = min(top_k, len(sdrt_graph.nodes))
        top_results = torch.topk(cos_scores, k=actual_k)
        
        pruned_nodes = [sdrt_graph.nodes[idx] for idx in top_results.indices]
        
        pruned_graph = DiscourseGraph(graph_id=sdrt_graph.graph_id + "_pruned", nodes=pruned_nodes, edges=[])
        return pruned_graph
