"""
Multi-Agent Communication Module

Implements communication mechanisms for multi-agent coordination:
- Message encoding from agent states
- Message aggregation (mean pooling, attention)
- Communication networks for information sharing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional


class CommunicationNetwork(nn.Module):
    """
    Neural communication network for multi-agent systems.
    
    Allows agents to share information through learned message passing.
    Supports both mean pooling and attention-based aggregation.
    
    Args:
        state_dim: Dimension of agent state
        message_dim: Dimension of communication messages
        hidden_dim: Hidden layer dimension
        use_attention: Whether to use attention for message aggregation
    """
    
    def __init__(
        self,
        state_dim: int = 5,
        message_dim: int = 16,
        hidden_dim: int = 32,
        use_attention: bool = False
    ):
        super().__init__()
        self.state_dim = state_dim
        self.message_dim = message_dim
        self.use_attention = use_attention
        
        # Message encoder: agent state -> message
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim)
        )
        
        # Message aggregator: messages -> aggregated message
        self.aggregator = nn.Sequential(
            nn.Linear(message_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, message_dim)
        )
        
        # Attention mechanism (optional)
        if use_attention:
            self.attention_query = nn.Linear(state_dim, message_dim)
            self.attention_key = nn.Linear(message_dim, message_dim)
            self.attention_value = nn.Linear(message_dim, message_dim)
    
    def encode_message(self, agent_state: torch.Tensor) -> torch.Tensor:
        """
        Generate message from agent state.
        
        Args:
            agent_state: Tensor of shape (batch, state_dim)
            
        Returns:
            message: Tensor of shape (batch, message_dim)
        """
        return self.encoder(agent_state)
    
    def aggregate_messages(
        self,
        messages: List[torch.Tensor],
        query_state: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Aggregate messages from multiple agents.
        
        Args:
            messages: List of message tensors, each (batch, message_dim)
            query_state: Optional query state for attention (batch, state_dim)
            
        Returns:
            aggregated: Tensor of shape (batch, message_dim)
        """
        if len(messages) == 0:
            # Return zero message if no messages
            return torch.zeros(1, self.message_dim, device=next(self.parameters()).device)
        
        # Stack messages: (n_agents, batch, message_dim)
        stacked = torch.stack(messages)
        
        if self.use_attention and query_state is not None:
            # Attention-based aggregation
            return self._aggregate_with_attention(stacked, query_state)
        else:
            # Mean pooling aggregation
            return self._aggregate_mean_pooling(stacked)
    
    def _aggregate_mean_pooling(self, messages: torch.Tensor) -> torch.Tensor:
        """
        Aggregate messages using mean pooling.
        
        Args:
            messages: Tensor of shape (n_agents, batch, message_dim)
            
        Returns:
            aggregated: Tensor of shape (batch, message_dim)
        """
        # Mean over agents dimension
        combined = messages.mean(dim=0)
        return self.aggregator(combined)
    
    def _aggregate_with_attention(
        self,
        messages: torch.Tensor,
        query_state: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate messages using attention mechanism.
        
        Args:
            messages: Tensor of shape (n_agents, batch, message_dim)
            query_state: Tensor of shape (batch, state_dim)
            
        Returns:
            aggregated: Tensor of shape (batch, message_dim)
        """
        # Compute attention query from agent state
        query = self.attention_query(query_state)  # (batch, message_dim)
        
        # Compute keys and values from messages
        # messages: (n_agents, batch, message_dim)
        n_agents, batch, _ = messages.shape
        messages_flat = messages.reshape(-1, self.message_dim)
        
        keys = self.attention_key(messages_flat).reshape(n_agents, batch, self.message_dim)
        values = self.attention_value(messages_flat).reshape(n_agents, batch, self.message_dim)
        
        # Compute attention scores: query * keys^T
        # query: (batch, message_dim) -> (batch, 1, message_dim)
        # keys: (n_agents, batch, message_dim) -> (batch, n_agents, message_dim)
        query = query.unsqueeze(1)  # (batch, 1, message_dim)
        keys = keys.permute(1, 0, 2)  # (batch, n_agents, message_dim)
        values = values.permute(1, 0, 2)  # (batch, n_agents, message_dim)
        
        # Attention weights
        scores = torch.matmul(query, keys.transpose(1, 2))  # (batch, 1, n_agents)
        scores = scores / (self.message_dim ** 0.5)  # Scale
        attention_weights = F.softmax(scores, dim=-1)  # (batch, 1, n_agents)
        
        # Weighted sum of values
        attended = torch.matmul(attention_weights, values)  # (batch, 1, message_dim)
        attended = attended.squeeze(1)  # (batch, message_dim)
        
        return self.aggregator(attended)
    
    def forward(
        self,
        agent_states: List[torch.Tensor],
        agent_idx: int
    ) -> torch.Tensor:
        """
        Complete communication forward pass for one agent.
        
        Args:
            agent_states: List of state tensors for all agents
            agent_idx: Index of the agent receiving messages
            
        Returns:
            aggregated_message: Tensor of shape (batch, message_dim)
        """
        # Encode messages from all agents
        messages = [self.encode_message(state) for state in agent_states]
        
        # Remove self-message (agent doesn't receive its own message)
        other_messages = [m for i, m in enumerate(messages) if i != agent_idx]
        
        # Aggregate messages
        if self.use_attention:
            return self.aggregate_messages(other_messages, agent_states[agent_idx])
        else:
            return self.aggregate_messages(other_messages)


class BroadcastCommunication:
    """
    Simple broadcast communication where all agents share all information.
    Useful baseline for comparison with learned communication.
    """
    
    def __init__(self, state_dim: int = 5):
        self.state_dim = state_dim
    
    def communicate(
        self,
        agent_states: List[torch.Tensor],
        agent_idx: int
    ) -> torch.Tensor:
        """
        Broadcast all other agents' states to current agent.
        
        Args:
            agent_states: List of state tensors for all agents
            agent_idx: Index of the agent receiving messages
            
        Returns:
            concatenated_states: All other agents' states concatenated
        """
        other_states = [s for i, s in enumerate(agent_states) if i != agent_idx]
        if len(other_states) == 0:
            return torch.zeros(1, self.state_dim, device=agent_states[0].device)
        return torch.cat(other_states, dim=-1)


class CommChannel:
    """
    Communication channel with bandwidth constraints and noise.
    More realistic communication model.
    """
    
    def __init__(
        self,
        message_dim: int = 16,
        bandwidth: int = None,  # Max messages per step
        noise_std: float = 0.0  # Gaussian noise std
    ):
        self.message_dim = message_dim
        self.bandwidth = bandwidth
        self.noise_std = noise_std
    
    def transmit(
        self,
        messages: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Transmit messages through channel with constraints.
        
        Args:
            messages: List of message tensors
            
        Returns:
            transmitted_messages: Messages after bandwidth/noise constraints
        """
        # Apply bandwidth constraint
        if self.bandwidth is not None and len(messages) > self.bandwidth:
            # Keep only first `bandwidth` messages (could use priority)
            messages = messages[:self.bandwidth]
        
        # Add noise
        if self.noise_std > 0:
            noisy_messages = []
            for msg in messages:
                noise = torch.randn_like(msg) * self.noise_std
                noisy_messages.append(msg + noise)
            return noisy_messages
        
        return messages


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("Testing Communication Networks...\n")
    
    # Test basic communication network
    print("1. Basic CommunicationNetwork (mean pooling):")
    comm_net = CommunicationNetwork(state_dim=5, message_dim=16, hidden_dim=32)
    
    # Simulate 3 agents
    agent_states = [
        torch.randn(1, 5),  # Agent 0
        torch.randn(1, 5),  # Agent 1
        torch.randn(1, 5),  # Agent 2
    ]
    
    # Agent 0 receives messages from others
    message = comm_net(agent_states, agent_idx=0)
    print(f"   Agent 0 received message shape: {message.shape}")
    print(f"   ✓ Mean pooling communication works\n")
    
    # Test attention-based communication
    print("2. CommunicationNetwork with attention:")
    comm_net_attn = CommunicationNetwork(
        state_dim=5,
        message_dim=16,
        hidden_dim=32,
        use_attention=True
    )
    
    message_attn = comm_net_attn(agent_states, agent_idx=0)
    print(f"   Agent 0 received message shape: {message_attn.shape}")
    print(f"   ✓ Attention-based communication works\n")
    
    # Test broadcast communication
    print("3. BroadcastCommunication:")
    broadcast = BroadcastCommunication(state_dim=5)
    broadcast_msg = broadcast.communicate(agent_states, agent_idx=0)
    print(f"   Agent 0 received broadcast shape: {broadcast_msg.shape}")
    print(f"   ✓ Broadcast communication works\n")
    
    # Test communication channel
    print("4. CommChannel with constraints:")
    channel = CommChannel(message_dim=16, bandwidth=2, noise_std=0.1)
    messages = [torch.randn(1, 16) for _ in range(5)]
    transmitted = channel.transmit(messages)
    print(f"   Original messages: {len(messages)}, Transmitted: {len(transmitted)}")
    print(f"   ✓ Communication channel works\n")
    
    print("✅ All communication tests passed!")
