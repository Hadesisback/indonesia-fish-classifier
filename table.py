from sklearn.metrics import classification_report, confusion_matrix
import json
import pandas as pd
import numpy as np
#main_dir='hasil'

with open("Resnet50_NoDataAug.txt",'r') as f:
	hasil=json.load(f)
	#print(hasil)
a=(classification_report(hasil[1],hasil[0],output_dict=True))
labels=list(set(hasil[1]))
df=pd.DataFrame(a).transpose()

#df.to_csv("mobilenet.csv")

df = pd.DataFrame(
    data=confusion_matrix(hasil[1],hasil[0], labels=labels),
    columns=labels,
    index=labels)

print(df[4])
print(df)
#
# Local (metrics per class)
#
tps = {}
fps = {}
fns = {}
tns = {}
specificity_local={}
precision_local = {}
recall_local = {}
f1_local = {}
accuracy_local = {}
for label in labels:
    tps[label] = df.loc[label, label]
    fps[label] = df[label].sum() - tps[label]
    fns[label] = df.loc[label].sum() - tps[label]
    tns[label] = len(hasil[1]) - (tps[label] + fps[label] + fns[label])
    tp, fp, fn, tn = tps[label], fps[label], fns[label], tns[label]
    print("ATNNN:",tn)
    precision_local[label] = tp / (tp + fp) if (tp + fp) > 0. else 0.
    recall_local[label] = tp / (tp + fn) if (tp + fp) > 0. else 0.
    p, r = precision_local[label], recall_local[label]
    specificity_local[label]= tn / (tn + fp) if (tn + fn) > 0. else 0.
    f1_local[label] = 2. * p * r / (p + r) if (p + r) > 0. else 0.
    accuracy_local[label] = tp / (tp + fp + fn) if (tp + fp + fn) > 0. else 0.

# print("#-- Local measures --#")
# print("True Positives:", tps)
# print("False Positives:", fps)
# print("False Negatives:", fns)
# print("Precision:", precision_local)
# print("Recall:", recall_local)
# print("F1-Score:", f1_local)
# print("Accuracy:", accuracy_local)

#
# Global
#
micro_averages = {}
macro_averages = {}

correct_predictions = sum(tps.values())
den = sum(list(tps.values()) + list(fps.values()))
micro_averages["Precision"] = 1. * correct_predictions / den if den > 0. else 0.

den = sum(list(tps.values()) + list(fns.values()))
micro_averages["Recall/Sensitivity"] = 1. * correct_predictions / den if den > 0. else 0.

micro_avg_p, micro_avg_r = micro_averages["Precision"], micro_averages["Recall/Sensitivity"]
micro_averages["F1-score"] = 2. * micro_avg_p * micro_avg_r / (micro_avg_p + micro_avg_r) if (micro_avg_p + micro_avg_r) > 0. else 0.

macro_averages["Precision"] = np.mean(list(precision_local.values()))
macro_averages["Recall/Sensitivity"] = np.mean(list(recall_local.values()))
macro_averages["Specificity"] = np.mean(list(specificity_local.values()))

macro_avg_p, macro_avg_r = macro_averages["Precision"], macro_averages["Recall/Sensitivity"]
macro_averages["F1-Score"] = 2. * macro_avg_p * macro_avg_r / (macro_avg_p + macro_avg_r) if (macro_avg_p + macro_avg_r) > 0. else 0.

total_predictions = df.values.sum()
accuracy_global = correct_predictions / total_predictions if total_predictions > 0. else 0.


import json
#tns = {}
#for label in set(hasil[1]):
    
# for label in labels:
#     tns[label] = len(hasil[1]) - (tps[label] + fps[label] + fns[label])
    
wrong=sum(tns.values())
den = sum(list(tns.values()) + list(fps.values()))
micro_averages["Specificity"] = 1. * wrong / den if den > 0. else 0.

print("#-- Global measures --#")
print("Micro-Averages:", micro_averages)
print("Macro-Averages:", macro_averages)
print("Correct predictions:", correct_predictions)
print("Total predictions:", total_predictions)
print("Accuracy:", accuracy_global)
print("True Negatives:", tns)

datas=pd.DataFrame(precision_local.items(),columns=["Class","Precision"])
datas['Recall']=datas['Class'].map(recall_local)
datas['F1-Score']=datas['Class'].map(f1_local)
datas['Accuracy']=datas['Class'].map(accuracy_local)
datas['Recall/Sensitivity']=datas['Class'].map(recall_local)
datas['Specificity']=datas['Class'].map(specificity_local)
datas.to_csv("InceptionV3.csv")
# datas=pd.DataFrame(precision_local.items(),columns=["Class","Precision"])
# datas['Recall']=datas['Class'].map(recall_local)
# datas['F1-Score']=datas['Class'].map(f1_local)
# datas['Accuracy']=datas['Class'].map(accuracy_local)
# datas['Recall']=datas['Class'].map(recall_local)


#print(datas)
