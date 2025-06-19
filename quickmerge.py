import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Gumbel
import numpy as np
from typing import Tuple, List, Optional
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity


class EntropyAwareSaliency(nn.Module):
    """Step 1: Multi-Scale Entropy-Aware Saliency Estimation"""
    
    def __init__(self, embedding_dim: int, num_layers: int = 12):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
    def compute_attention_entropy(self, X: torch.Tensor) -> torch.Tensor:
        """Compute attention entropy across layers"""
        B, N, D = X.shape
        entropy_scores = torch.zeros(B, N, device=X.device)
        
        # Simulate multi-layer attention computation
        for layer in range(self.num_layers):
            # Compute attention matrix
            attention_scores = torch.matmul(X, X.transpose(-2, -1)) / np.sqrt(D)
            attention_probs = F.softmax(attention_scores, dim=-1)
            
            # Compute entropy for each token
            entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-8), dim=-1)
            entropy_scores += entropy
            
        # Average across layers and normalize
        entropy_scores = entropy_scores / self.num_layers
        normalized_entropy = (entropy_scores - entropy_scores.min(dim=-1, keepdim=True)[0]) / \
                           (entropy_scores.max(dim=-1, keepdim=True)[0] - entropy_scores.min(dim=-1, keepdim=True)[0] + 1e-8)
        
        # Lower entropy = higher saliency
        saliency = 1.0 - normalized_entropy
        return saliency
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.compute_attention_entropy(X)


class DifferentiableTokenMerging(nn.Module):
    """Step 2: Differentiable Saliency-Guided Token Merging"""
    
    def __init__(self, temperature: float = 0.1, epsilon: float = 0.01):
        super().__init__()
        self.temperature = temperature
        self.epsilon = epsilon
        
    def gumbel_softmax_selection(self, saliency: torch.Tensor) -> torch.Tensor:
        """Select salient tokens using Gumbel-Softmax"""
        gumbel_dist = Gumbel(0, 1)
        gumbel_noise = gumbel_dist.sample(saliency.shape).to(saliency.device)
        
        logits = (saliency + gumbel_noise) / self.temperature
        mask = F.softmax(logits, dim=-1)
        
        return mask
    
    def cluster_tokens(self, X: torch.Tensor, saliency_mass: torch.Tensor, K: int) -> Tuple[torch.Tensor, List[List[int]]]:
        """Cluster tokens using cosine similarity"""
        B, N, D = X.shape
        merged_tokens = torch.zeros(B, K, D, device=X.device)
        token_groups = []
        
        for b in range(B):
            # Compute cosine similarity matrix
            X_norm = F.normalize(X[b], p=2, dim=-1)
            similarity_matrix = torch.matmul(X_norm, X_norm.transpose(-2, -1))
            
            # Use agglomerative clustering
            clustering = AgglomerativeClustering(n_clusters=K, linkage='average')
            cluster_labels = clustering.fit_predict(similarity_matrix.cpu().numpy())
            
            # Group tokens by cluster
            groups = [[] for _ in range(K)]
            for i, label in enumerate(cluster_labels):
                groups[label].append(i)
            
            # Compute merged tokens via saliency-weighted averaging
            for k, group in enumerate(groups):
                if group:
                    weights = saliency_mass[b, group]
                    weights = weights / (weights.sum() + 1e-8)
                    merged_tokens[b, k] = torch.sum(X[b, group] * weights.unsqueeze(-1), dim=0)
            
            token_groups.append(groups)
        
        return merged_tokens, token_groups
    
    def forward(self, X: torch.Tensor, saliency: torch.Tensor, K: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate binary mask via Gumbel-Softmax
        mask = self.gumbel_softmax_selection(saliency)
        
        # Compute saliency mass
        saliency_mass = mask * saliency + (1 - mask) * self.epsilon
        
        # Cluster and merge tokens
        merged_tokens, token_groups = self.cluster_tokens(X, saliency_mass, K)
        
        return merged_tokens, mask


class BidirectionalARAlignment(nn.Module):
    """Step 3: Autoregressive Prior Alignment"""
    
    def __init__(self, embedding_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Forward AR model
        self.forward_ar = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Backward AR model
        self.backward_ar = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward_loss(self, X: torch.Tensor) -> torch.Tensor:
        """Compute forward autoregressive loss"""
        B, K, D = X.shape
        loss = 0.0
        
        for t in range(K - 1):
            context = X[:, :t+1].mean(dim=1)  # Simplified context aggregation
            pred = self.forward_ar(context)
            target = X[:, t+1]
            loss += F.mse_loss(pred, target)
        
        return loss / (K - 1)
    
    def backward_loss(self, X: torch.Tensor) -> torch.Tensor:
        """Compute backward autoregressive loss"""
        B, K, D = X.shape
        loss = 0.0
        
        for t in range(K-1, 0, -1):
            context = X[:, t:].mean(dim=1)  # Simplified context aggregation
            pred = self.backward_ar(context)
            target = X[:, t-1]
            loss += F.mse_loss(pred, target)
        
        return loss / (K - 1)
    
    def compute_alignment_loss(self, X: torch.Tensor) -> torch.Tensor:
        """Compute bidirectional AR alignment loss"""
        forward_loss = self.forward_loss(X)
        backward_loss = self.backward_loss(X)
        return forward_loss + backward_loss


class NormBasedFidelityConstraint(nn.Module):
    """Step 4: Norm-Based Fidelity Constraint"""
    
    def __init__(self, gamma: float = 0.8):
        super().__init__()
        self.gamma = gamma
    
    def compute_fidelity_loss(self, X: torch.Tensor, X_merged: torch.Tensor) -> torch.Tensor:
        """Compute fidelity constraint loss"""
        B, N, D = X.shape
        K = X_merged.shape[1]
        
        # Compute norm-mass retention ratio
        token_norms = torch.norm(X, p=2, dim=-1)  # [B, N]
        total_norm = torch.sum(token_norms, dim=-1, keepdim=True)  # [B, 1]
        
        # Find top-K tokens by norm
        top_k = min(K, N // 2)
        top_norms, _ = torch.topk(token_norms, k=top_k, dim=-1)
        top_norm_sum = torch.sum(top_norms, dim=-1, keepdim=True)
        
        gamma_actual = top_norm_sum / total_norm
        
        # Pad merged sequence to original length
        X_merged_pad = F.pad(X_merged, (0, 0, 0, N - K), value=0.0)
        
        # Compute reconstruction loss
        reconstruction_loss = F.mse_loss(X, X_merged_pad, reduction='none')
        reconstruction_loss = torch.mean(reconstruction_loss, dim=-1)  # [B]
        
        # Fidelity constraint
        fidelity_threshold = (1 - gamma_actual.squeeze()) ** 2 * torch.norm(X, p='fro', dim=(-2, -1)) ** 2
        
        # Penalty for violating constraint
        violation_penalty = F.relu(reconstruction_loss - fidelity_threshold)
        
        return torch.mean(violation_penalty)


class QuickMergePP(nn.Module):
    """Main QuickMerge++ Framework"""
    
    def __init__(self, embedding_dim: int, num_layers: int = 12, temperature: float = 0.1, 
                 gamma: float = 0.8, epsilon: float = 0.01):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.saliency_estimator = EntropyAwareSaliency(embedding_dim, num_layers)
        self.token_merger = DifferentiableTokenMerging(temperature, epsilon)
        self.ar_alignment = BidirectionalARAlignment(embedding_dim)
        self.fidelity_constraint = NormBasedFidelityConstraint(gamma)
    
    def forward(self, X: torch.Tensor, K: int, alpha: float = 0.1) -> Tuple[torch.Tensor, dict]:
        """
        Forward pass of QuickMerge++
        
        Args:
            X: Input embeddings [B, N, D]
            K: Target number of tokens
            alpha: AR alignment weight
            
        Returns:
            merged_tokens: Compressed sequence [B, K, D]
            losses: Dictionary of losses
        """
        B, N, D = X.shape
        
        # Step 1: Compute saliency scores
        saliency = self.saliency_estimator(X)
        
        # Step 2: Differentiable token merging
        merged_tokens, mask = self.token_merger(X, saliency, K)
        
        # Step 3: AR alignment loss
        ar_loss = self.ar_alignment.compute_alignment_loss(merged_tokens)
        
        # Step 4: Fidelity constraint loss
        fidelity_loss = self.fidelity_constraint.compute_fidelity_loss(X, merged_tokens)
        
        # Combine losses
        total_loss = alpha * ar_loss + fidelity_loss
        
        losses = {
            'ar_alignment': ar_loss.item(),
            'fidelity': fidelity_loss.item(),
            'total': total_loss.item()
        }
        
        return merged_tokens, losses
    
    def inference(self, X: torch.Tensor, K: int, ar_model=None) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Inference pipeline for QuickMerge++
        
        Args:
            X: Input embeddings [N, D]
            K: Target number of tokens
            ar_model: Autoregressive model for decoding
            
        Returns:
            merged_tokens: Compressed sequence [K, D]
            predictions: List of predicted tokens
        """
        self.eval()
        with torch.no_grad():
            # Add batch dimension
            X_batch = X.unsqueeze(0)  # [1, N, D]
            
            # Compute saliency
            saliency = self.saliency_estimator(X_batch)
            
            # Generate mask
            mask = self.token_merger.gumbel_softmax_selection(saliency)
            saliency_mass = mask * saliency + (1 - mask) * self.token_merger.epsilon
            
            # Cluster and merge tokens
            merged_tokens, _ = self.token_merger.cluster_tokens(X_batch, saliency_mass, K)
            merged_tokens = merged_tokens.squeeze(0)  # [K, D]
            
            # Perform autoregressive decoding if model provided
            predictions = []
            if ar_model is not None:
                ar_model.eval()
                with torch.no_grad():
                    for t in range(K):
                        if t == 0:
                            pred = ar_model(merged_tokens[:1])
                        else:
                            pred = ar_model(merged_tokens[:t+1])
                        predictions.append(pred[-1])
            
            return merged_tokens, predictions


def quickmerge_inference(X: torch.Tensor, ar_model, entropy_budget: float = 0.5) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """
    QuickMerge++ Inference Algorithm
    
    Args:
        X: Token embeddings [N, D]
        ar_model: Autoregressive generation model
        entropy_budget: Compression ratio (K = N * entropy_budget)
        
    Returns:
        merged_tokens: Compressed sequence
        predictions: Autoregressive predictions
    """
    N, D = X.shape
    K = max(1, int(N * entropy_budget))
    
    # Initialize QuickMerge++
    quickmerge = QuickMergePP(embedding_dim=D)
    
    # Perform inference
    merged_tokens, predictions = quickmerge.inference(X, K, ar_model)
    
    return merged_tokens, predictions 