# Version 1 of the environment
from gym import Env, spaces
from gym.spaces import Discrete, Box
import numpy as np

class BiasEnv_v1(Env):
    def __init__(self):
        # Actions we can take, price range from 5 to 10
        self.size = 6
        self.action_space = Box(low=5, high=10, shape=(self.size,))
        # voltage & power usage array
        self.observation_space = spaces.Dict(
            {
                "load": Box(low=0, high=20, shape=(self.size,), dtype=int),
                "voltage": Box(low=0.95, high=1.05, shape=(self.size,), dtype=float)
            }
        ) 
        # was Box(low=np.array(np.ones((2,6))*0), high=np.array(np.ones((2,6)))*[[20]*6,[2]*6])
        # Set start state, assume that every nodes have random load
        # self.state = self.observation_space.sample()#np.ones((1,6))[0]*5 + np.random.randint(0,6,size=6)
        # Set length 100 time steps
        self.sim_length = 100
    def _get_obs(self):
        return {"load": self._load, "voltage": self._voltage}

    def step(self, action):
        # Apply action
        self._load = self.load_response(action)
        
        # Reduce simulation length by 1
        self.sim_length -= 1
        
        # Calculate reward
        self._voltage, v_cost = self.voltage_cost()
        reward = action@self._load + v_cost
        
        # Check if simulation is done
        if self.sim_length <= 0:
            done = True
        else:
            done = False
        
        observation = self._get_obs()
        # Set placeholder for info
        info = v_cost
        
        # Return step information
        return observation, reward, done, info
    def load_response(self, action):
        # Caluclate load according to prices y = -x + 15
        u_load = np.zeros((1, len(action)))[0]
        for i, price in enumerate(action):
            u_load[i] = max(5,min(10,(-price + 15)))
            # if abs(u_load[i] - self.state[i])>1:
            #     u_load[i] = np.sign(u_load[i] - self.state[i]) * 1 + self.state[i]
        return u_load
    def voltage_cost(self):
        total_load = sum(self._load)
        voltage = np.ones((1,6))[0]
        v_cost = 0
        if sum(self._load)<40:
            return voltage,v_cost
        else:
            for i in range(0,len(self._load)):
                seed = np.random.randint(0,5+i)
                if seed>3:
                    voltage[i] = (self._voltage[i]-5+1)*0.1+1
                    v_cost -= 20
            return voltage, v_cost
        
        
    
    def render(self):
        # Implement viz
        pass
    def reset(self):
        # Reset state
        self._load = np.ones((1,6))[0]*5 + np.random.randint(0,6,size=6)
        # Set length 100 time steps
        self._voltage = np.ones((1,6))[0]
        self.sim_length = 100
        done = False
        observation = self._get_obs
        return observation

if __name__ == '__main__':
    env = BiasEnv_v1()
    print(env.action_space.sample())
    print(env.observation_space.sample())

    episodes = 10
    for episode in range(1, episodes+1):
        state = env.reset()
        done = False
        score = 0
        
        while not done:
            action = env.action_space.sample()
            n_state, reward, done, info = env.step(action)
            score += reward
        print("Episode:{} Score:{}".format(episode, score))