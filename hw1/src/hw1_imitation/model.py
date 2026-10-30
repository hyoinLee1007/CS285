"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from math import tau
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    ### TODO: IMPLEMENT MSEPolicy HERE ###
    def __init__(
        self,
        state_dim: int, # 5
        action_dim: int, # 2
        chunk_size: int, 
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("hidden_dims must contain positive integers.")
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, chunk_size * action_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).reshape(-1, self.chunk_size, self.action_dim)


    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:

        predicted_actions = self.forward(state)
        loss = nn.MSELoss()(predicted_actions, action_chunk)
        return loss

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        sampled_actions = self.forward(state)
        return sampled_actions
    


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    ### TODO: IMPLEMENT FlowMatchingPolicy HERE ###
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        layers = []
        input_dim = action_dim * chunk_size + state_dim + 1 # A_tau, tau, S

        for hidden_dim in hidden_dims:
            if hidden_dim <= 0:
                raise ValueError("hidden_dims must contain positive integers.")
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, chunk_size * action_dim))
        self.net = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:

        # A0, tau Sampling
        a1 = action_chunk.flatten(start_dim=1)
        a0 = torch.randn_like(a1)
        tau = torch.rand(
            (state.shape[0], 1),
            device=a1.device,
            dtype=a1.dtype,
        )

        # A_tau
        a_tau = (1 - tau) * a0 + tau * a1
        input_tensor = torch.cat([a_tau, state, tau], dim=-1)
        predicted_velocity = self.net(input_tensor)

        target_velocity = a1 - a0
        return nn.functional.mse_loss(predicted_velocity, target_velocity)

    def sample_actions( # inference 시에만 사용됨
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        # Batchsize
        batch_size = state.shape[0]

        # A0 sampling (32,16) / action_dim : 2 / chunk_size : 8
        n = self.action_dim*self.chunk_size
        A0 = torch.randn(batch_size, n,
                         device=state.device,
                        dtype=state.dtype,
                )
        
        # 추론 시, tau = 0 초기화
        tau = state.new_full((batch_size, 1), 0.0)
        dt = 1.0 / num_steps
        a_tau = A0
        
        for _ in range(num_steps):
            
            input_tensor = torch.cat([a_tau, state, tau], dim=-1)

            # 현재 속도 예측
            velocity = self.net(input_tensor)

            # 액션과 시간 갱신
            a_tau = a_tau + dt * velocity
            tau = tau + dt

        return a_tau.reshape(
            batch_size, self.chunk_size, self.action_dim
        )

PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
