# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 02:44:06 2025

@author: there
"""

#import pandas as pd

import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.python.client import device_lib
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.metrics import ConfusionMatrixDisplay
    




#NN training
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


model = Sequential([
    Dense(4096, activation='relu', input_shape=(39,)),  BatchNormalization(),Dropout(0.3),

    Dense(2048, activation='relu'),BatchNormalization(),Dropout(0.3),

    Dense(1024, activation='relu'),  BatchNormalization(),Dropout(0.3),

    Dense(512, activation='relu'), BatchNormalization(),Dropout(0.2),

    Dense(256, activation='relu'),  BatchNormalization(),Dropout(0.2),

    Dense(8, activation='softmax')  
])


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# List of .pkl files
file_paths = ["dfPPP_1.pkl", "dfPPP_2.pkl", "dfPPP_3.pkl", "dfPPP_4.pkl", "dfPPP_5.pkl", "dfPPP_6.pkl", "dfPPP_7.pkl"]


batch_size = 32
epochs = 5  
for file_path in file_paths:
    print(f"Loading {file_path}...")
    df = pd.read_pickle(f'/home/yashchonkar33/cyberWise/data/{file_path}')
    #change labels to groups
    original_one_hot_columns = df.columns[39:]
    df['Group_Label'] = df[original_one_hot_columns].idxmax(axis=1).map(attack_mapping)
    group_one_hot = pd.get_dummies(df['Group_Label'])
    df = df.drop(columns=original_one_hot_columns)
    df = pd.concat([df, group_one_hot], axis=1)
    df = df.drop(columns=['Group_Label'])
    
    # Split into features and labels
    features = df.iloc[:, :39].values 
    labels = df.iloc[:, 39:].values  
    del df
    # Train model the current file
    model.fit(features, labels, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    del features
    del labels
    print(f'{file_path} done FINALLY')


model.save("/home/yashchonkar33/cyberWise/modelGroup.h5")
print("Model training complete. Saved as /home/yashchonkar33/cyberWise/modelGroup.h5")
#save model

test_df = pd.read_pickle('/home/yashchonkar33/cyberWise/data/dfPPP_8.pkl')
#groups test labels
original_one_hot_columns = test_df.columns[39:]
test_df['Group_Label'] = test_df[original_one_hot_columns].idxmax(axis=1).map(attack_mapping)
group_one_hot = pd.get_dummies(test_df['Group_Label'])
test_df = test_df.drop(columns=original_one_hot_columns)
test_df = pd.concat([test_df, group_one_hot], axis=1)
test_df = test_df.drop(columns=['Group_Label'])

#generate predictions on test set
X_test = test_df.iloc[:, :39].values  
y_test = test_df.iloc[:, 39:].values  
test_df = None
predictions = model.predict(X_test, batch_size=1024)  
predicted_classes = predictions.argmax(axis=1)  
true_classes = y_test.argmax(axis=1)  

accuracy = (true_classes == predicted_classes).mean() 
print(f"Test Accuracy: {accuracy * 100:.2f}%")

labels = ['Benign', 'Brute-Force', 'DDoS', 'DoS', 'Mirai', 'Recon', 'Spoofing', 'Web-Attacks']  
cm = confusion_matrix(true_classes, predicted_classes, labels=range(len(labels)))

#Display and save confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title('Confusion Matrix')
plt.savefig('/home/yashchonkar33/cyberWise/confusion_matrix.png')
print("Confusion matrix saved to /home/yashchonkar33/cyberWise/confusion_matrix.png")





