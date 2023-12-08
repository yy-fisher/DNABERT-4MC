
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import torch
import torch.nn as nn
import numpy as np

def Model_Evaluate(confus_matrix):
    TN, FP, FN, TP = confus_matrix.ravel()
    print(TN, FP, FN, TP)

    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    MCC = ((TP * TN) - (FP * FN)) / (np.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)))

    print("Model score --- SN:{0:<20}SP:{1:<20}ACC:{2:<20}MCC:{3:<20}\n".format(SN, SP, ACC, MCC))

    return SN, SP, ACC, MCC

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 由于inputs是字典类型的，定义一个辅助函数帮助to(device)
def to_device(dict_tensors):
    result_tensors = {}
    for key, value in dict_tensors.items():
        result_tensors[key] = value.to(device)
    return result_tensors


def cal_score(pred, label):
    pred = np.around(pred)
    label = np.array(label)
    confus_matrix = confusion_matrix(label, pred, labels=None, sample_weight=None)
    SN, SP, ACC, MCC = Model_Evaluate(confus_matrix)

    return ACC
def fit(model, train_loader, optimizer, criterion, device):
    model.train()

    pred_list = []
    label_list = []

    for inputs, targets ,seq_feature in tqdm(train_loader):
        inputs, targets,seq_feature = to_device(inputs), targets.to(device),seq_feature.to(device)
        outputs = model(inputs,seq_feature)
#         print(outputs.shape,"fu")

        loss = criterion(outputs.squeeze(), targets.float())
        loss.backward()
        optimizer.step()
        model.zero_grad()

        pred_list.extend(outputs.squeeze().cpu().detach().numpy())
        label_list.extend(targets.squeeze().cpu().detach().numpy())
    train_score = cal_score(pred_list, label_list)
#     print(train_score)
    return train_score
