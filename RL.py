import numpy as np
import pandas as pd
import gym
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns
# CyberSecurity Environment
class CyberSecurityEnv(gym.Env):
    def __init__(self):
        super(CyberSecurityEnv, self).__init__()
        #only relevant features will be part of env
        self.features = ['Time_To_Live', 'syn_flag_number', 'rst_flag_number', 'ack_count', 'syn_count', 'fin_count', 'HTTP', 'HTTPS', 'DNS', 'SSH', 'UDP', 'ARP', 'Tot sum', 'Number']
        
        #data for enviornemnt
        self.attack_df = pd.DataFrame() 
        self.benign_df = pd.DataFrame()
        #enviornment vars
        self.state_size = len(self.features)
        self.action_size = 12  
        self.max_steps = 50  
        # Action and state space
        self.action_space = gym.spaces.Discrete(self.action_size)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(self.state_size,), dtype=np.float32)

    def reset(self):
        #SETTING UP ENV
        self.action_number = 0
        self.step_count = 0
        #state is feature vector of random attack, target is a random benign state vector
        attack_state = self.attack_df.sample(1).iloc[0]
        benign_state = self.benign_df.sample(1).iloc[0]
        self.attack_type = attack_state['Group_Label']
        self.state = attack_state[self.features].values.astype(np.float32)  
        self.benign_target = benign_state[self.features].values.astype(np.float32) 

        return self.state

    def step(self, action):
        self.step_count += 1
        #Actions(commented) and the effect thet will have on features
        action_map = {
    0: {'HTTP': -0.3, 'Number': -0.01},  # Rate limit HTTP 
    1: {'HTTPS': -0.3, 'Number': -0.01},  # Rate limit HTTPS 
    2: {'Time_To_Live': +5, 'Number': -0.01},  # Increase TTL 
    3: {'syn_flag_number': -0.3, 'syn_count': -0.3, 'Number': -0.01},  # Rate limit SYN 
    4: {'ack_flag_number': -0.3, 'ack_count': -0.3, 'Number': -0.01},  # Rate limit ACK 
    5: {'rst_flag_number': -.3, 'Number': -0.01},  #Rate limit RST
    6: {'UDP': -0.3, 'Number': -0.01},  # Rate limit UDP 
    7: {'ARP': -0.3, 'Number': -0.01},  # Rate limit ARP 
    8: {'Tot sum': -0.3, 'Number': -0.01},  # Rate limit oversized packets
    9: {'fin_count': -0.3, 'Number': -0.01},  # Rate limit FIN
    10: {'SSH': -0.3, 'Number': -0.01},  # Rate limit SSH 
    11: {'DNS': -0.3, 'Number': -0.01},  # Rate limit DNS
}

        # execute action
        for feature, change in action_map[action].items():
            if feature in self.features:
                idx = self.features.index(feature)
                self.state[idx] =  self.state[idx] + change
        
        
        self.action_number += 1 
        

        # reward function is based on difference between benign and current state
        diff = np.abs(self.state - self.benign_target)
        #penalize number of actions needed
        action_penalty = self.action_number * 0.05
        reward = -np.sum(diff) - action_penalty  
        done = np.sum(diff) < 5 or (self.step_count >= self.max_steps) 
        return self.state.astype(np.float32), reward, done, {}

# Deep Q Network  Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=500)
        
        # Hyperparams
        self.discount = 0.5 
        self.explore = .5  
        self.explore_min = 0.01
        self.explore_decay = 0.8
        self.learning_rate = 0.2
        self.batch_size = 32

        self.model = self.build_model()

    def build_model(self):
        #Neural net that represents models policy descision
        model = Sequential([
            Dense(128, input_dim=self.state_size, activation='relu'),  
            Dense(128, activation='relu'),  
            Dense(64, activation='relu'),   
            Dense(64, activation='relu'),   
            Dense(self.action_size, activation='linear') ])
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        #add expereince
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        #act randomly
        if np.random.rand() <= self.explore:
            return random.randrange(self.action_size)  
        #get prediction for next action
        q_values = self.model.predict(np.array([state], dtype=np.float32), verbose=0)
        return np.argmax(q_values[0])  
    
    def replay(self):
        #train NN for agents policy making
        if len(self.memory) < self.batch_size:
            return 
        
        indices = np.random.choice(len(self.memory), self.batch_size, replace=False)
        minibatch = [self.memory[i] for i in indices]
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.discount * np.amax(self.model.predict(np.array([next_state], dtype=np.float32), verbose=0)[0])

            target_f = self.model.predict(np.array([state], dtype=np.float32), verbose=0)
            target_f[0][action] = target

            self.model.fit(np.array([state], dtype=np.float32), target_f, epochs=1, verbose=0)

        # lower explore 
        if self.explore > self.explore_min:
            self.explore *= self.explore_decay


# Training Loop
file_paths = [ "/home/yashchonkar33/cyberWise/data/dfPPP_1.pkl","/home/yashchonkar33/cyberWise/data/dfPPP_2.pkl","/home/yashchonkar33/cyberWise/data/dfPPP_3.pkl","/home/yashchonkar33/cyberWise/data/dfPPP_4.pkl","/home/yashchonkar33/cyberWise/data/dfPPP_5.pkl","/home/yashchonkar33/cyberWise/data/dfPPP_6.pkl","/home/yashchonkar33/cyberWise/data/dfPPP_7.pkl"]
  

#Create env and agent obj
env = CyberSecurityEnv()
agent = DQNAgent(env.state_size, env.action_size)
#used to processes df and extract analysis
attack_mapping = {
    "Label_DDOS-ACK_FRAGMENTATION": "DDoS",
    "Label_DDOS-HTTP_FLOOD": "DDoS",
    "Label_DDOS-ICMP_FLOOD": "DDoS",
    "Label_DDOS-ICMP_FRAGMENTATION": "DDoS",
    "Label_DDOS-PSHACK_FLOOD": "DDoS",
    "Label_DDOS-RSTFINFLOOD": "DDoS",
    "Label_DDOS-SLOWLORIS": "DDoS",
    "Label_DDOS-SYNONYMOUSIP_FLOOD": "DDoS",
    "Label_DDOS-SYN_FLOOD": "DDoS",
    "Label_DDOS-TCP_FLOOD": "DDoS",
    "Label_DDOS-UDP_FLOOD": "DDoS",
    "Label_DDOS-UDP_FRAGMENTATION": "DDoS",
    
    "Label_DOS-HTTP_FLOOD": "DoS",
    "Label_DOS-SYN_FLOOD": "DoS",
    "Label_DOS-TCP_FLOOD": "DoS",
    "Label_DOS-UDP_FLOOD": "DoS",
    
    "Label_MIRAI-GREETH_FLOOD": "Mirai",
    "Label_MIRAI-GREIP_FLOOD": "Mirai",
    "Label_MIRAI-UDPPLAIN": "Mirai",
    
    "Label_RECON-HOSTDISCOVERY": "Recon",
    "Label_RECON-OSSCAN": "Recon",
    "Label_RECON-PINGSWEEP": "Recon",
    "Label_RECON-PORTSCAN": "Recon",
    "Label_VULNERABILITYSCAN": "Recon",
    
    "Label_SQLINJECTION": "Web-Attacks",
    "Label_XSS": "Web-Attacks",
    "Label_UPLOADING_ATTACK": "Web-Attacks",
    "Label_BROWSERHIJACKING": "Web-Attacks",
    "Label_COMMANDINJECTION": "Web-Attacks",
    "Label_BACKDOOR_MALWARE": "Web-Attacks",
    
    "Label_DICTIONARYBRUTEFORCE": "Brute-Force",
    
    "Label_DNS_SPOOFING": "Spoofing",
    "Label_MITM-ARPSPOOFING": "Spoofing",
    
    "Label_BENIGN": "Benign"
}
episodes = 1500  
count = 0
# keep track of stats for insights
action_usage_per_attack = {}
rewards_per_attack = {}
for file in file_paths:
    print(f"\n Switching to dataset: {file}")
    
    #Load new dataset into environment
    df = pd.read_pickle(file)
    original_one_hot_columns = df.columns[39:]
    df['Group_Label'] = df[original_one_hot_columns].idxmax(axis=1).map(attack_mapping)
    df = df.drop(columns=original_one_hot_columns)
    #create two different dfs so sampling is not expensive
    attack_df = df[df['Group_Label'] != 'Benign']
    benign_df = df[df['Group_Label'] == 'Benign']
    
    
    
    env.attack_df = attack_df
    env.benign_df = benign_df
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        attack_label = env.attack_type
        total_reward = 0
        
        if attack_label not in action_usage_per_attack:
            action_usage_per_attack[attack_label] = np.zeros(env.action_size)
            rewards_per_attack[attack_label] = []

        while not done:
            action = agent.act(state)  
            action_usage_per_attack[attack_label][action] += 1
            next_state, reward, done, _ = env.step(action) 
            agent.remember(state, action, reward, next_state, done)  
            state = next_state
            total_reward += reward
        agent.replay()
        rewards_per_attack[attack_label].append(total_reward)  
        if episode % 10 == 0:
            print(f'{episode}s done')
            
            
#PRINT STATS
print("DONE TRAINING \n")
print('Now printing stats \n')

average_rewards_per_attack = {
attack: sum(rewards) / len(rewards) if rewards else 0  
for attack, rewards in rewards_per_attack.items()}

print(average_rewards_per_attack)


action_distribution_per_attack = {
attack: (actions / actions.sum() if actions.sum() > 0 else np.zeros_like(actions))
for attack, actions in action_usage_per_attack.items()}

print(action_distribution_per_attack)



#MAKE VISUALIZATIONS

plt.figure(figsize=(20, 12))

sns.barplot(x=list(average_rewards_per_attack.keys()), y=list(average_rewards_per_attack.values()))
plt.title('Average Reward per Attack Type')
plt.xlabel('Attack Type')
plt.ylabel('Average Reward')
plt.xticks(rotation=45, ha='right') 
plt.tight_layout()  
plt.savefig('/home/yashchonkar33/cyberWise/average_rewards_per_attack.png') 

plt.close()

#ORINGAL *uncommented RUN GAVE ERROR FOR PLOT CREATION, SO PLOT CREATION WAS DONE AFTERWARD FROM PLOT CREATION
'''
action_distributions = np.array(list(action_distribution_per_attack.values()))
actions = np.arange(env.action_size)  
plt.figure(figsize=(12, 7))
plt.bar(actions, action_distributions.T, label=list(action_distribution_per_attack.keys()), stacked=True)
plt.title('Action Distribution per Attack Type')
plt.xlabel('Action')
plt.ylabel('Distribution (Percentage)')
plt.legend(title='Attack Type', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.xticks(actions)
plt.tight_layout()


plt.savefig('/home/yashchonkar33/cyberWise/action_distribution_per_attack.png') 
plt.close()
'''
#ACTION DIST ACROSS CLASSES, MADE USING PRINTED DICTIONARY FROM ORIGINAL TRAIN
attack_probabilities = {
    'DDoS': np.array([0.075, 0.085, 0.0016, 0.015, 0.116, 0.096, 0.100, 0.068, 0.183, 0.058, 0.099, 0.098]),
    'DoS': np.array([0.067, 0.083, 0.0014, 0.023, 0.134, 0.091, 0.090, 0.056, 0.190, 0.050, 0.102, 0.107]),
    'Recon': np.array([0.045, 0.096, 0.0007, 0.030, 0.123, 0.089, 0.110, 0.051, 0.177, 0.104, 0.089, 0.081]),
    'Mirai': np.array([0.054, 0.079, 0.0007, 0.018, 0.131, 0.107, 0.118, 0.060, 0.169, 0.054, 0.098, 0.104]),
    'Spoofing': np.array([0.083, 0.056, 0.0000, 0.018, 0.158, 0.120, 0.037, 0.046, 0.186, 0.047, 0.149, 0.093]),
    'Web-Attacks': np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.4, 0.0, 0.004, 0.0, 0.2, 0.0, 0.196])
}


action_labels = [
    "Rate limit HTTP", "Rate limit HTTPS", "Increase TTL", "Rate limit SYN", "Rate limit ACK",
    "Rate limit RST", "Rate limit UDP", "Rate limit ARP", "Rate limit oversized packets",
    "Rate limit FIN", "Rate limit SSH", "Rate limit DNS"
]
colors = {
    'DDoS': 'red',
    'DoS': 'blue',
    'Recon': 'green',
    'Mirai': 'purple',
    'Spoofing': 'orange',
    'Web-Attacks': 'brown'
}
plt.figure(figsize=(12, 6))

for attack, probabilities in attack_probabilities.items():
    plt.plot(action_labels, probabilities, marker='o', label=attack, color=colors[attack])

plt.xticks(rotation=45, ha='right')
plt.xlabel("Actions Taken")
plt.ylabel("Probability")
plt.title("Attack Type Probabilities Over Different Actions")
plt.legend()
plt.grid(True)

# Show plot
plt.tight_layout()
plt.show()
