import rlpg.models.common as common
from typing import Tuple

import torch

class ActorModel (torch.nn.Module):
    def sample (self, params: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def distribution (self, params) -> torch.distributions.Distribution:
        pass

class ActorModelContinuous (ActorModel):
    def __init__ (
        self,
        observation_sz: int,
        action_sz: int,
        std_clamp: Tuple[float, float] = (-20,2)
    ):
        super().__init__()

        self._std_clamp = std_clamp
        self.action_size = action_sz
        self.observation_size = observation_sz
        self.actor = common.LinearBlock([
            observation_sz,
            observation_sz * 20,
            observation_sz * 5
        ], torch.relu)
        self.mu_layer = torch.nn.Linear(self.actor.out_size, action_sz)
        self.std_layer = torch.nn.Linear(self.actor.out_size, action_sz)

    def forward (self, state):
        # Predict a new action
        act_res = self.actor(state)
        mu = self.mu_layer(act_res)
        std = self.std_layer(act_res)
        # log_std = torch.clamp(log_std, self._std_clamp[0], self._std_clamp[1])
        std = torch.exp(std) # ensure the values are positive. torch.abs would break the math
        return torch.stack([mu, std],dim=1)
    
    def distribution (self, params):
        mu, sigma = params[:,0,:], params[:,1,:]
        dist = torch.distributions.Normal(mu, sigma)
        return dist
    
    def sample (self, params):
        dist = self.distribution(params)
        action = dist.sample()
        return action, dist.log_prob(action)

class ActorModelDiscrete (ActorModel):
    def __init__(self, observation_sz: int, action_sz: int) -> None:
        super().__init__()
        self.action_size = action_sz
        self.observation_size = observation_sz
        self.actor = common.LinearBlock([
            observation_sz,
            observation_sz * 20,
            observation_sz * 5,
            action_sz 
        ], torch.relu)

    def forward (self, state):
        # Predict a new action
        act_res = self.actor(state).view(-1, self.action_size)
        return act_res
    
    def distribution (self, params):
        dist = torch.distributions.Categorical(logits=params)
        return dist

    def sample (self, params):
        dist = self.distribution(params)
        action = dist.sample()
        return action, dist.log_prob(action)
    
# class CriticModel (torch.nn.Module):
#     def __init__(self, observation_sz, action_sz) -> None:
#         super().__init__()

#         # TODO: This might need to be shared with the actor...
#         self.state = common.LinearBlock([
#             observation_sz,
#             observation_sz * 4,
#             observation_sz * 8
#         ], torch.relu)

#         self.action = common.LinearBlock([
#             action_sz,
#             self.state.out_size
#         ], torch.relu)

#         self.q_predictor = common.LinearBlock([
#             self.state.out_size,
#             observation_sz * 4,
#             1
#         ], torch.relu)
#         self.observation_sz = observation_sz
#         self.action_sz = action_sz

#     def forward (self, state, action):

#         # Predict QValue for a state and action
#         st_res = self.state(state)
#         a_res = self.action(action)
#         merged = st_res + a_res
#         qval = self.q_predictor(merged)

#         return qval

# class CriticModelDiscrete (CriticModel):
#     def forward (self, state):
#         avec = torch.nn.functional.one_hot(action, self.action_sz).float()
#         # Predict QValue for a state and action
#         st_res = self.state(state)
#         a_res = self.action(avec)
#         merged = st_res + a_res
#         qval = self.q_predictor(merged)

#         return qval

class CriticModel (torch.nn.Module):
    def __init__(self, observation_size, action_size) -> None:
        super().__init__()
        self.critic = common.LinearBlock([
            observation_size,
            observation_size * 8,
            observation_size * 2,
            1
        ], torch.relu)
        self.observation_size = observation_size
        self.action_size = action_size

    def forward (self, state):
        score = self.critic(state).squeeze()
        return score



def init_actor_critic_continuous (observation_sz: int, action_sz: int, **kwargs):
    return ActorModelContinuous(observation_sz, action_sz), CriticModel(observation_sz, action_sz)

def init_actor_critic_discrete (observation_sz: int, action_sz: int, **kwargs):
    return ActorModelDiscrete(observation_sz, action_sz), CriticModel(observation_sz, action_sz)

