import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class DeepQConvNetwork(nn.Module):
    def __init__(self, lr, input_dims, hidden_dim, n_actions):
        super(DeepQConvNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.hidden_dim = hidden_dim

        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, stride = 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride = 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride = 1)
        self.pool = nn.AvgPool2d(kernel_size = 3, stride = 2)
        self.fc1 = nn.Linear(2304, self.hidden_dim)
        # self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc2 = nn.Linear(self.hidden_dim, self.n_actions)


        # Xavier initialization.
        def init_xavier(m):
            if type(m) in (nn.Linear, nn.Conv2d):
                torch.nn.init.xavier_uniform_(m.weight)
            
        self.apply(init_xavier)

        # DQN estimates the value of each action given a set of states.
        self.optimizer = optim.Adam(self.parameters(), lr = lr, amsgrad = True)
        self.loss = nn.MSELoss()

        # add replay memory.
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

        print('Initializing QNet - device : ', self.device)

    def forward(self, state):
        # print('state', state.shape)
        state = state.permute(0, 3, 1, 2)
        # print('after', state.shape)
        x = F.elu(self.conv1(state)) # unsqueeze 1 else
        #print(1, x.shape)
        x = F.elu(self.conv2(x))
        # print('2', x.shape)
        x = F.elu(self.pool(self.conv3(x)))
        # print('out0', x.shape)
        x = x.reshape(x.shape[0], -1)
        # print('out1', x.shape)
        
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        return x

    def decay_lr(self, factor = .5):
        print('decaying learning rate - factor = 0.5.')
        for g in self.optimizer.param_groups:
            g['lr'] = g['lr']*factor


# class DeepQNetwork(nn.Module):
#     def __init__(self, lr, input_dims, hidden_dim, n_actions):
#         super(DeepQNetwork, self).__init__()
#         self.input_dims = input_dims
#         self.n_actions = n_actions
#         self.hidden_dim = hidden_dim

        
#         self.fc1 = nn.Linear(np.prod(input_dims), self.hidden_dim)
#         self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.fc3 = nn.Linear(self.hidden_dim, self.n_actions)


#         # Xavier initialization.
#         def init_xavier(m):
#             if type(m) in (nn.Linear, nn.Conv2d):
#                 torch.nn.init.xavier_uniform_(m.weight)
            
#         self.apply(init_xavier)

#         # DQN estimates the value of each action given a set of states.
#         self.optimizer = optim.Adam(self.parameters(), lr = lr, amsgrad = True)
#         self.loss = nn.MSELoss()

#         # add replay memory.
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.to(self.device)

#         print('Initializing QNet - device : ', self.device)

#     def forward(self, state):
#         # print('state', state.shape)
#         state = state.view(state.shape[0], -1)
#         # print('after', state.shape)
#         x = F.relu(self.fc1(state)) # unsqueeze 1 else
#         #print(1, x.shape)
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         return x

#     def decay_lr(self, factor = .5):
#         print('decaying learning rate - factor = 0.5.')
#         for g in self.optimizer.param_groups:
#             g['lr'] = g['lr']*factor


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(np.prod(self.input_dims), self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        # DQN estimates the value of each action given a set of states.
        self.optimizer = optim.Adam(self.parameters(), lr = lr)
        self.loss = nn.MSELoss()

        # add replay memory.
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state.reshape(state.shape[0], -1)))
        x = F.relu(self.fc2(x))
        return self.fc3(x) #actions
    

    
# Model free bootstrapped off_policy env
class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, 
        batch_size, n_actions, max_mem_size = 100000, eps_end = 0.01, eps_dec = 5e-4):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_end
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims, fc1_dims = 512, fc2_dims = 256)
        # self.Q_eval = DeepQConvNetwork(lr = self.lr, input_dims=input_dims, hidden_dim=512, n_actions=n_actions)
        ## self.Q_eval = DeepQNetwork(lr = self.lr, input_dims=input_dims, hidden_dim=512, n_actions=n_actions)
        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype = np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype = np.float32)

        self.action_memory = np.zeros(self.mem_size, dtype = np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype = np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype = np.bool)

    def get_state(self, observation):
        # print('observation', observation.shape)
        state = (observation.astype(np.float32) - 1)/2
        state = state# .flatten()
        return state

    def store_transition(self, observation, action, reward, observation_, done):
        index = self.mem_cntr%self.mem_size

        state = self.get_state(observation)
        state_ = self.get_state(observation_)

        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor(self.get_state(observation)).to(self.Q_eval.device)[None]
            actions = self.Q_eval.forward(state)
            action = torch.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action
    
    def save(self, path='models/dqn.pt'):
        torch.save(self.Q_eval, path)

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return 
        
        self.Q_eval.optimizer.zero_grad()
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace = False)
        
        batch_index = np.arange(self.batch_size, dtype = np.int32)

        state_batch = torch.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)

        action_batch = self.action_memory[batch]

        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma*torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
