from .lib.env_base import *
from .libs.traj_util import *
from .environment import *

def load_envs(env_id, path="./"):
	return pickle_load(path+"/"+env_id)

def onehot_to_index(onehot):
	return np.argmax(np.array(onehot))

def vec_to_action(vec):
	if vec==[0,-1]:
		return onehot(0, 4)
	elif vec==[-1,0]:
		return onehot(1, 4)
	elif vec==[0,1]:
		return onehot(2, 4)
	elif vec==[1,0]:
		return onehot(3, 4)

class MultiGridWorldEnv():
	def __init__(self, state_size):
		agents = create_airl_env(state_size)
		self.env = create_env(agents)
		self.state = [self.e.start_pos for e in range(self.env)]

    def step(action_list):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}

        next_state = [0 for _ in range(self.num_agents)] 
        for i in range(self.num_agents):
        	if has_done(state[i]): 
        		next_state[i] = state[i]
        	else:
        		act = onehot_to_index(action_list[i])
		        next_state[i] = self.env[i]._move(state[i], act)
	        reward_n.append(self.env[i].get_reward(next_state[i]))
	        done_n.append(has_done(next_state[i]))
	        obs_n.append(next_state[i])
    	return obs_n, reward_n, done_n, info_n

    @property
    def num_envs(self):
        return 1
    @property
    def state_size(self):
        return self.env[0].gred
    @property
    def num_obs(self):
        return 1
    @property
    def num_actions(self):
        return self.env[0].num_actions
    @property
    def num_states(self):
        return self.env[0].num_states
    @property
    def num_agents(self):
        return len(env)
    @property
    def observation_space(self):
        return [e.observation_space for e in range(self.env)]
    @property
    def action_space(self):
        return [e.action_space for e in range(self.env)]
