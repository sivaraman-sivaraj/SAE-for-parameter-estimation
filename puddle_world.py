from gym import Env
from gym.envs.registration import register
from gym.utils import seeding
from gym import spaces
import numpy as np

class PuddleWorld(Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 60
    }
    
    def __init__(self, wind=True, end=0):
        '''
        The initialization of the grid-world parameters.

        Parameters
        ----------
        wind : Boolean value, True if the wind is on otherwise False.
        end : 0, 1 or 2 corresponding to the end goals A, B or C respectively.

        Returns
        -------
        None.

        '''
        self.wind = wind
        self.end = end
        
        self.actions = np.array([0,1,2,3]) # up, down, left, right
        self.rewards = np.zeros((12,12))
        self.goals = np.array([(0, 11), (2,9), (6,7)])
        self.start = np.array([(5,0), (6,0), (10,0), (11,0)])
        
        puddle1 = []
        for i in range(5):
            puddle1.append((2,3+i))
            self.rewards[2][3+i] = -1
            puddle1.append((8,3+i))
            self.rewards[8][3+i] = -1
        puddle1.append((2,8))
        self.rewards[2][8] = -1
        for i in range(4):
            puddle1.append((3+i,3))
            self.rewards[3+i][3] = -1
            puddle1.append((3+i,8))
            self.rewards[3+i][8] = -1
        puddle1.append((7,3))
        self.rewards[7][3] = -1
        puddle1.append((6,7))
        self.rewards[6][7] = -1
        puddle1.append((7,7))
        self.rewards[7][7] = -1
        self.puddle1 = np.array(puddle1)
        
        puddle2 = []
        for i in range(3):
            puddle2.append((3,4+i))
            self.rewards[3][4+i] = -2
            puddle2.append((7,4+i))
            self.rewards[7][4+i] = -2
        puddle2.append((3,7))
        self.rewards[3][7] = -2
        for i in range(2):
            puddle2.append((4+i,4))
            self.rewards[4+i][4] = -2
            puddle2.append((4+i,7))
            self.rewards[4+i][7] = -2
        puddle2.append((6,4))
        self.rewards[6][4] = -2
        puddle2.append((5,6))
        self.rewards[5][6] = -2
        puddle2.append((6,6))
        self.rewards[6][6] = -2
        self.puddle2 = np.array(puddle2)
        
        puddle3 = []
        puddle3.append((4,5))
        self.rewards[4][5] = -3
        puddle3.append((5,5))
        self.rewards[5][5] = -3
        puddle3.append((6,5))
        self.rewards[6][5] = -3
        puddle3.append((4,6))
        self.rewards[4][6] = -3
        self.puddle3 = np.array(puddle3)
        
        self.Goal = self.goals[end]
        self.rewards[self.Goal[0]][self.Goal[1]] = 10
        
        self.done = False
        self.seed()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def step(self, action):
        self.next_state = np.copy(self.current_state)
        
        if(np.random.rand() >= 0.9):
            tmp = np.random.randint(0,4)
            if(tmp >= action):
                action = tmp+1
        
        if(action==0):
            self.next_state[0] = max(self.current_state[0]-1, 0)
        elif(action==1):
            self.next_state[0] = min(self.current_state[0]+1, 11)
        elif(action==2):
            self.next_state[1] = max(self.current_state[1]-1, 0)
        elif(action==3):
            self.next_state[1] = min(self.current_state[1]+1, 11)
                
        if(self.wind):
            if(np.random.rand() < 0.5):
                self.next_state[1] = min(self.next_state[1]+1, 11)
            
        reward = self.rewards[self.next_state[0]][self.next_state[1]]
        self.current_state = np.copy(self.next_state)
        if((self.current_state == self.Goal).all()):
            self.done = True
        
        return self.current_state, reward
    
    def reset(self):
        self.current_state = self.start[np.random.randint(0,4)]
        self.done = False
    
    def render(self, mode='human', close=False):
        pass
    