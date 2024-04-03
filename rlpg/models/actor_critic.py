import rlpg.models.common as common

import torch

class ActorModel (torch.nn.Module):
    def __init__ (
        self,
        observation_sz: int,
        action_sz: int
    ):
        super().__init__()

        self.action_size = action_sz
        self.observation_size = observation_sz
        self.actor = common.LinearBlock([
            observation_sz,
            observation_sz * 30,
            observation_sz * 20,
            observation_sz * 10,
            observation_sz * 2,
            action_sz * 2 # TODO: try having the sigma learned as well..
        ], torch.relu)

    def forward (self, state):
        # Predict a new action
        act_res = self.actor(state).view(-1, 2, self.action_size)
        # act_res = self.actor(state).view(-1, self.action_size)
        return act_res
    
    def distribution (self, params):
        mu, sigma = params[:,0,:], params[:,1,:]
        dist = torch.distributions.Normal(mu, torch.abs(sigma))
        # dist = torch.distributions.Normal(params, 0.1)
        return dist

    
    def sample (self, params):
        dist = self.distribution(params)
        action = dist.sample()
        return action, dist.log_prob(action)
    

class CriticModel (torch.nn.Module):
    def __init__(self, observation_sz, action_sz) -> None:
        super().__init__()

        # TODO: This might need to be shared with the actor...
        self.state = common.LinearBlock([
            observation_sz,
            observation_sz * 4,
            observation_sz * 8
        ], torch.relu)

        self.action = common.LinearBlock([
            action_sz,
            self.state.out_size
        ], torch.relu)

        self.q_predictor = common.LinearBlock([
            self.state.out_size,
            observation_sz * 4,
            1
        ], torch.relu)

    def forward (self, state, action):

        # Predict QValue for a state and action
        st_res = self.state(state)
        a_res = self.action(action)
        merged = st_res + a_res
        qval = self.q_predictor(merged)

        return qval

def init_actor_critic (observation_sz: int, action_sz: int, **kwargs):
    return ActorModel(observation_sz, action_sz), CriticModel(observation_sz, action_sz)

