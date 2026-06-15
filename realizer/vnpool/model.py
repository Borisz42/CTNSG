import torch
import torch.nn as nn

from .decoder import RVQDecoder
from .pooling import PerceiverIOPooling
from .projector import LLMProjector
from .injector import TextualizationInjector

class VNPoolRealizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int = 64,
        d_gvt: int = 256,
        num_quantizers: int = 4,
        num_latents: int = 8,
        n_heads: int = 8,
        d_llm: int = 4096,
        llm_model: nn.Module = None
    ):
        """
        Master module integrating the RVQ Decoder, Perceiver IO Pooler, and LLM Projector.
        Configures the provided LLM with heavily calibrated PEFT LoRA adapters.
        """
        super().__init__()
        
        self.decoder = RVQDecoder(
            num_embeddings=num_embeddings, 
            embedding_dim=d_gvt, 
            num_quantizers=num_quantizers
        )
        
        self.pooler = PerceiverIOPooling(
            in_dim=d_gvt, 
            num_latents=num_latents, 
            latent_dim=d_gvt, 
            n_heads=n_heads
        )
        
        self.projector = LLMProjector(
            in_dim=d_gvt, 
            out_dim=d_llm
        )
        
        self.injector = TextualizationInjector()
        
        self.llm = llm_model
        
        # Configure LoRA if LLM is provided
        if self.llm is not None:
            self._configure_lora()
            
    def _configure_lora(self):
        """
        Injects tightly calibrated LoRA adapters into the LLM.
        Explicitly enforces r=16, alpha=32, dropout=0.05 on q_proj and v_proj.
        """
        from peft import LoraConfig, get_peft_model
        
        # Target Qwen's standard attention projections
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        self.llm = get_peft_model(self.llm, lora_config)
        # Ensure LoRA adapters are actively trained
        self.llm.train()
        print("[VNPool] Successfully configured and activated LoRA (r=16, a=32, d=0.05) on the LLM backbone.")

    def forward(self, discrete_indices: torch.Tensor, text_graph_emb: torch.Tensor, query_emb: torch.Tensor):
        """
        Full Realizer Pipeline: Decode -> Pool -> Project -> Inject
        
        Args:
            discrete_indices: Hierarchical RVQ indices [batch_size, num_nodes, N_RVQ]
            text_graph_emb: Textual graph embeddings [batch_size, seq_len_graph, D_LLM]
            query_emb: User query embeddings [batch_size, seq_len_query, D_LLM]
        Returns:
            hybrid_embeddings: [batch_size, 8 + seq_len_graph + seq_len_query, D_LLM]
        """
        # 1. Decode RVQ tokens to continuous vectors (sums over N_RVQ)
        z_q = self.decoder(discrete_indices)
        
        # 2. Compress via Perceiver IO (outputs exactly K=8 tokens)
        pooled = self.pooler(z_q)
        
        # 3. Project to LLM dimensions
        projected = self.projector(pooled)
        
        # 4. Inject with textual scaffold
        hybrid_prompt = self.injector.inject(projected, text_graph_emb, query_emb)
        
        return hybrid_prompt

if __name__ == '__main__':
    # --- Local Verification Test ---
    print("Executing VNPool Realizer verification test...")
    torch.manual_seed(42)
    
    # Dimensions
    batch_size = 2
    num_nodes = 45 # Variable graph size
    n_rvq = 4
    d_gvt = 256
    d_llm = 4096 # e.g., Qwen-3.5-4B
    
    # 1. Mock the discrete tokens coming from RelDiT/GVT
    # Codebook size K=64
    discrete_indices = torch.randint(0, 64, (batch_size, num_nodes, n_rvq))
    
    # 2. Mock the textual scaffold embeddings from the LLM
    text_graph_emb = torch.randn((batch_size, 15, d_llm))
    query_emb = torch.randn((batch_size, 10, d_llm))
    
    # 3. Mock the LLM backbone
    # We will use a simple linear layer to simulate the LLM's classification head for gradient checking
    class MockLLM(nn.Module):
        def __init__(self):
            super().__init__()
            # Simulate q_proj and v_proj for PEFT to hook into
            self.q_proj = nn.Linear(d_llm, d_llm)
            self.v_proj = nn.Linear(d_llm, d_llm)
            self.classifier = nn.Linear(d_llm, 100)
            self.config = type('Config', (), {'model_type': 'qwen2'})()
            
        def forward(self, inputs_embeds, **kwargs):
            x = self.q_proj(inputs_embeds) + self.v_proj(inputs_embeds)
            # Pool to simulate sequence classification
            x = x.mean(dim=1) 
            return self.classifier(x)
            
        def prepare_inputs_for_generation(self, *args, **kwargs):
            pass
            
    mock_llm = MockLLM()
    
    # 4. Instantiate VNPool Realizer
    vnpool = VNPoolRealizer(
        num_embeddings=64,
        d_gvt=d_gvt,
        num_quantizers=n_rvq,
        num_latents=8,
        d_llm=d_llm,
        llm_model=mock_llm
    )
    
    # Ensure gradients are enabled for the projector and pooler
    vnpool.train()
    
    # 5. Forward Pass
    print(f"\n--- Forward Dimensionality Verification ---")
    hybrid_embeddings = vnpool(discrete_indices, text_graph_emb, query_emb)
    print(f"Hierarchical Tokens Input Shape: {discrete_indices.shape}")
    print(f"Hybrid Prompt Output Shape: {hybrid_embeddings.shape}")
    
    # Check dimensionality: 8 (VNPool) + 15 (Text Graph) + 10 (Query) = 33 tokens
    assert hybrid_embeddings.shape == (batch_size, 33, d_llm), "Shape mismatch in hybrid prompt!"
    
    # 6. Backward Pass Gradient Flow Verification
    print(f"\n--- Backward Gradient Verification ---")
    
    # Feed hybrid embeddings through the LLM
    outputs = vnpool.llm(inputs_embeds=hybrid_embeddings)
    loss = outputs.sum()
    loss.backward()
    
    # Check if gradients successfully propagated backward into the projector
    proj_has_grad = False
    for param in vnpool.projector.parameters():
        if param.grad is not None:
            proj_has_grad = True
            break
            
    # Check if gradients successfully propagated backward into the pooler
    pooler_has_grad = False
    if vnpool.pooler.latent_queries.grad is not None:
        pooler_has_grad = True
        
    print(f"Gradient flowed to LLM Projector: {proj_has_grad}")
    print(f"Gradient flowed to Perceiver Pooler: {pooler_has_grad}")
    
    assert proj_has_grad and pooler_has_grad, "Gradient flow broken!"
    
    print("\n[SUCCESS] VNPool dimensionality reduction, scaffold injection, and gradient flow verified.")
