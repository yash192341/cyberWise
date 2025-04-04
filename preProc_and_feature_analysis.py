"""
Created on Thu Mar 27 15:29:54 2025

@author: there
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns


#USED FOR CIC IOT 2023 DATASET, BREAK UP DATA INTO 7 DFS
#extract each csv from  pathname and load into several dataframes(to manage ram) stored using pickle, save to save_path(folder)
def load_csv_to_df(pathname,save_path): 
    csv_list = []
    count = 0
    pckl_num = 1
    for csv in os.listdir(pathname):
        csv_path = os.path.join(pathname,csv)
        chunk_size = 10000
        chunk_list = []
        for chunk in pd.read_csv(csv_path, chunksize=chunk_size):
            chunk_list.append(chunk)
        dfc = pd.concat(chunk_list, ignore_index=True)
        csv_list.append(dfc)
        dfc = None
        print(f'Finished {csv}')
        count += 1
        if(count == 7):
            df = pd.concat(csv_list,axis=0)
            df.reset_index(drop=True, inplace=True)
            save_path = save_path + f'df_{pckl_num}.pkl'
            with open(save_path,'wb') as file:
                pickle.dump(df,file)
            count = 0
            pckl_num += 1
            csv_list = []
            df = None



#pass in path to file to load 
def load_pkl(path):
    with open(path,'rb') as file:
        return pickle.load(file)

#preprocess df for uses
def preprocessing(df,idn):
    #drop missing
    df.dropna(inplace=True)
    
    #drop all inf values in rate
    df = df[~df['Rate'].isin([np.inf, -np.inf])]
    '''
    one_hot_labels = pd.get_dummies(df['Label'], prefix='Label')
    df = pd.concat([df.drop(columns=['Label']), one_hot_labels], axis=1)
    '''
    #scale data
    scaler = StandardScaler()
    numeric_columns = df.columns[:39]
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    df.reset_index(drop=True, inplace=True)
    #UNCOMMENT TO PROCESSES AGAIN FOR CLASSIFIER  TRAINING(uncomment attack mapping for RL too)
    '''
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
    original_one_hot_columns = df.columns[39:]
    df['Group_Label'] = df[original_one_hot_columns].idxmax(axis=1).map(attack_mapping)
    group_one_hot = pd.get_dummies(df['Group_Label'])
    df = df.drop(columns=original_one_hot_columns)
    df = pd.concat([df, group_one_hot], axis=1)
    df = df.drop(columns=['Group_Label'])
    '''
    
    #SWITCH BACK FOR RL
    '''
    original_one_hot_columns = df.columns[39:]
    df['Group_Label'] = df[original_one_hot_columns].idxmax(axis=1).map(attack_mapping)
    df = df.drop(columns=original_one_hot_columns)
    '''
    
    #save and print for debugging
    if(idn == 1):
        print(df.head())
        print(df.columns)
    with open('D:\\cyberWise\\processed_CIC\\' + f'dfPPP_{idn}.pkl','wb') as file:
        pickle.dump(df,file)
    print(f'Dataset {idn} pre processed')
        
    
    
    
    
    
#to load csvs to dataframes
load_csv_to_df('ENTER CSV  FOLDER PATH AS STRING HERE', 'ENTE SAVE PATH FOR DFS')


#preprocess dfs
for i in range(1,10):
    dfp = load_pkl('ENTER DF PATH HERE, ENSURE TO USE FOR LOOP VARIABLE TO SPECIFY WHICH ONE, OR CHANGE LOOP') #ex: 'D:\\cyberWise\\processed_CIC\\' + f'df_{i}.pkl'
    preprocessing(dfp, i)
    dfp = None





#build sampled df for intial feature analysis
sampled_df = pd.DataFrame()
for i in range(1,10):
    #REPLACE WITH YOUR LOCATION, CHANGE LOOP IF YOU HAVE TO
    dfp = load_pkl('D:\\cyberWise\\processed_CIC\\' + f'dfPPP_{i}.pkl')
    random_rows = dfp.sample(n=1_200_000, replace=False, random_state=42)
    dfp = None
    sampled_df = pd.concat([sampled_df, random_rows], ignore_index=True)
    random_rows = None
    print(f'finished {i}')
print(sampled_df)

with open('D:\\cyberWise\\processed_CIC\\sampled_df.pkl','wb') as file:
    pickle.dump(sampled_df,file)



#INTIAL FEATURE ANALYSIS
dfp = pickle.read('D:\\cyberWise\\processed_CIC\\sampled_df.pkl')
#HEATMAP CREATION
features = ['Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate',
       'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
       'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
       'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count', 'rst_count',
       'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP',
       'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max',
       'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Variance']

# Group by class (Label) and calculate mean values for each feature
mean_values = dfp.groupby('Label')[features].mean()

# Create the heatmap
plt.figure(figsize=(20, 12))  
sns.heatmap(mean_values, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, cbar_kws={'label': 'Mean Value'})
plt.title("Heatmap of Feature Contributions Across Classes")
plt.ylabel("Classes (Label)")
plt.xlabel("Features")
#save
#raise dpi based on your machine, heatmap is hard to read
plt.savefig("/home/yashchonkar33/cyberWise/vis/feature_heatmap.png", dpi=300)  
print("heatmap done")
plt.close()

#MUTUAL INFO
label_col = dfp["Label"]
dfp.drop(columns=["Label"], inplace=True)

mi_scores = mutual_info_classif(dfp, label_col, discrete_features=False)
dfp["Label"] = label_col
plt.figure(figsize=(40, 20)) 
plt.bar(dfp.columns[:-1], mi_scores)  # Skip "Label" in column names
plt.xlabel("Features")
plt.ylabel("Mutual Information Score")
plt.title("Feature Importance via Mutual Information")
plt.xticks(rotation=45)
plt.savefig("/home/yashchonkar33/cyberWise/vis/mtual_info.png", dpi=300)
print(mi_scores)










