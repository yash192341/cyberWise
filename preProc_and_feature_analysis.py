
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
    one_hot_labels = pd.get_dummies(df['Label'], prefix='Label')
    df = pd.concat([df.drop(columns=['Label']), one_hot_labels], axis=1)
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



'''
dfp = load_pkl('D:\\cyberWise\\processed_CIC\\' + f'dfPP_1.pkl')
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
plt.figure(figsize=(20, 12))  # Make the plot large enough to show all features and classes
sns.heatmap(mean_values, annot=True, fmt=".2f", cmap='coolwarm', linewidths=0.5, cbar_kws={'label': 'Mean Value'})
plt.title("Heatmap of Feature Contributions Across Classes")
plt.ylabel("Classes (Label)")
plt.xlabel("Features")

# Save heatmap to the "plots" folder
plt.savefig("plots/feature_contributions_heatmap.png", dpi=300)  # High resolution for clarity
plt.close()
'''


'''
stats = dfp.groupby('Label')[features].agg(['mean', 'std'])
for feature in features:
    plt.figure(figsize=(15, 8))  # Adjust figure size for readability
    
    # Extract mean and std for the feature across classes
    mean_values = stats[(feature, 'mean')]
    std_values = stats[(feature, 'std')]
    
    # Create the bar plot
    plt.errorbar(mean_values.index, mean_values.values, yerr=std_values.values, fmt='o', capsize=5, color='blue', label=f'{feature}')
    plt.xlabel("Classes (Label)")
    plt.ylabel(f"{feature} Value")
    plt.title(f"Mean and Standard Deviation of {feature} Across Classes")
    plt.xticks(ticks=mean_values.index, labels=mean_values.index, rotation=45)  
    
    # Save plot locally
    plt.savefig(f"plots/{feature}_mean_std_plot.png", dpi=300)  # High resolution ensures clarity
    plt.close()  # Close plot to free memory
'''
    


'''
dfp = load_pkl('D:\\cyberWise\\processed_CIC\\' + f'dfPP_1.pkl')
label_col = dfp["Label"]
dfp.drop(columns=["Label"], inplace=True)

mi_scores = mutual_info_classif(dfp[0:500000], label_col[0:500000], discrete_features=False)
dfp["Label"] = label_col

print(mi_scores)
#processed_CIC\\
#dfp = load_pkl('D:\\cyberWise\\CIC_TEST.pkl')
#print(dfp.columns)
'''


'''
dfp = dfp[~dfp['Rate'].isin([np.inf, -np.inf])]
labels_remove = ['BACKDOOR_MALWARE', 'BENIGN', 'BROWSERHIJACKING', 'COMMANDINJECTION',  'DICTIONARYBRUTEFORCE', 'DNS_SPOOFING', 'MIRAI-GREETH_FLOOD', 'MIRAI-GREIP_FLOOD', 'MIRAI-UDPPLAIN', 'SQLINJECTION', 'UPLOADING_ATTACK', 'VULNERABILITYSCAN', 'XSS']
dfp = dfp[~dfp['Label'].isin(labels_remove)]

label_encoder = LabelEncoder()
dfp['Label'] = label_encoder.fit_transform(dfp['Label'])
print("Label Mapping:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

numeric_columns = dfp.drop(columns=["Label"]).columns
dfp[numeric_columns] = scaler.fit_transform(dfp[numeric_columns])


label_col = dfp["Label"]
dfp.drop(columns=["Label"], inplace=True)

mi_scores = mutual_info_classif(dfp, label_col, discrete_features=False)
dfp["Label"] = label_col

plt.bar(dfp.columns[:-1], mi_scores)  # Skip "Label" in column names
plt.xlabel("Features")
plt.ylabel("Mutual Information Score")
plt.title("Feature Importance via Mutual Information")
plt.xticks(rotation=45)
plt.show()
print(mi_scores)
'''


'''
features = ['Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate',
       'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
       'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
       'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count', 'rst_count',
       'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP',
       'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max',
       'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Variance',]
mi_scores = [1.11839558e+00, 9.52806603e-01, 1.25657219e-01, 1.92531233e-01,
 3.47289203e-01, 6.03348258e-01, 3.76540439e-01, 3.80320439e-01,
 4.33234711e-01, 1.60431749e-03, 9.17957655e-04, 4.35811472e-01,
 5.96289233e-01, 3.43536469e-01, 3.73328862e-01, 6.16307686e-02,
 8.57908152e-02, 2.81472460e-02, 0.00000000e+00, 1.50036082e-04,
 2.28105366e-03, 9.22058472e-04, 7.72002328e-01, 6.07698664e-01,
 2.40955763e-03, 3.86089164e-02, 5.43139596e-01, 0.00000000e+00,
 3.80933617e-02, 3.80255610e-02, 2.98895762e-01, 1.14756079e-01,
 2.52650023e-01, 2.77177447e-01, 2.38753719e-01, 2.77079178e-01,
 1.94075422e-01, 7.90484477e-02, 2.37890089e-01]

plt.figure(figsize=(20, 10))
plt.bar(features, mi_scores, color='skyblue')  # Customize bar color if needed
plt.xlabel("Features")
plt.ylabel("Mutual Information Score")
plt.title("Feature Importance via Mutual Information")
plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility

# Save the plot locally with high resolution
plt.savefig("mutual_information_plot.png", dpi=300)  # Saves as PNG; adjust filename and format as needed
plt.close()  # Close the plot to release memory
'''



'''
features = ['Header_Length', 'Protocol Type', 'Time_To_Live', 'Rate',
       'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
       'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
       'cwr_flag_number', 'ack_count', 'syn_count', 'fin_count', 'rst_count',
       'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC', 'TCP', 'UDP',
       'DHCP', 'ARP', 'ICMP', 'IGMP', 'IPv', 'LLC', 'Tot sum', 'Min', 'Max',
       'AVG', 'Std', 'Tot size', 'IAT', 'Number', 'Variance']

mi_scores = [1.11652069e+00, 9.50196873e-01, 1.27764308e-01, 1.87308033e-01,  
3.47138603e-01, 6.00170599e-01, 3.73833805e-01, 3.79725881e-01,  
4.29989509e-01, 6.67517536e-04, 4.25121736e-04, 4.33521293e-01,  
5.92383631e-01, 3.41537097e-01, 3.70850346e-01, 6.15087272e-02,  
8.60270705e-02, 2.84312888e-02, 1.50986337e-03, 1.24368074e-03,  
2.24768133e-03, 1.29140636e-03, 7.71022516e-01, 6.06548487e-01,  
3.10892668e-03, 3.82159788e-02, 5.41791159e-01, 6.95498978e-04,  
3.83361587e-02, 3.85683759e-02, 3.00274330e-01, 1.15321718e-01,  
2.53677438e-01, 2.78136174e-01, 2.39197853e-01, 2.77678401e-01,  
1.68367198e-01, 7.92882575e-02, 2.36145616e-01]

plt.figure(figsize=(20, 6))
plt.bar(features, mi_scores, color='skyblue', edgecolor='black')
plt.xlabel("Features", fontsize=14)
plt.ylabel("Mutual Information Score", fontsize=14)
plt.title("Feature Importance Based on Mutual Information", fontsize=16)
plt.xticks(rotation=45, fontsize=12)
plt.tight_layout()

# Save the plot directly to the working directory
plt.savefig("mi_scores_plot.png", dpi=300)  # Save with high resolution
plt.close()  # Close the plot to avoid overwriting
'''
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










