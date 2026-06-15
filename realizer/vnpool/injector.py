import torch
import torch.nn as nn

class TextualizationInjector:
    def __init__(self):
        """
        Handles the concatenation of soft graph tokens with the textual scaffold
        to anchor the topological structure to semantic reality.
        """
        pass
        
    def inject(self, vn_embeddings: torch.Tensor, text_graph_embeddings: torch.Tensor, query_embeddings: torch.Tensor):
        """
        Concatenates the representations to prevent catastrophic hallucination.
        
        Args:
            vn_embeddings: Virtual Node Pool embeddings [batch_size, 8, D_LLM]
            text_graph_embeddings: LLM embeddings for explicit triples [batch_size, seq_len_graph, D_LLM]
            query_embeddings: LLM embeddings for user query [batch_size, seq_len_query, D_LLM]
            
        Returns:
            hybrid_prompt: [batch_size, 8 + seq_len_graph + seq_len_query, D_LLM]
        """
        # Strict order: [8_VN_Tokens, Textualized_Graph_Embeddings, Query_Embeddings]
        return torch.cat([vn_embeddings, text_graph_embeddings, query_embeddings], dim=1)
