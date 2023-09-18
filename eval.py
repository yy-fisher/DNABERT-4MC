import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
def Model_Evaluate(confus_matrix):
    TN, FP, FN, TP = confus_matrix.ravel()
    print(TN, FP, FN, TP)

    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    MCC = ((TP * TN) - (FP * FN)) / (np.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)))

    print("Model score --- SN:{0:<20}SP:{1:<20}ACC:{2:<20}MCC:{3:<20}\n".format(SN, SP, ACC, MCC))

    return SN, SP, ACC, MCC


from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve


def cal_AUC(prob, labels):
    fpr_1, tpr_1, threshold_1 = roc_curve(labels, prob)
    #     print(labels,"ww")
    #     roc_auc = auc(fpr_1,tpr_1)   # 准确率代表所有正确的占所有数据的比值
    #     print('roc_auc:', roc_auc)

    f = list(zip(prob, labels))
    rank = [values2 for values1, values2 in sorted(f, key=lambda x: x[0])]
    rankList = [i + 1 for i in range(len(rank)) if rank[i] == 1]
    posNum = 0
    negNum = 0
    for i in range(len(labels)):
        if (labels[i] == 1):
            posNum += 1
        else:
            negNum += 1
    auc_final = 0
    auc_final = (sum(rankList) - (posNum * (posNum + 1)) / 2) / (posNum * negNum)
    #     print(auc)
    plt.figure(figsize=(5, 5))

    plt.plot(fpr_1, tpr_1, color='darkorange', lw=2,
             label='ROC curve (area = %0.2f)' % auc_final)  ###假正率为横坐标，真正率为纵坐标做曲线

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])

    plt.ylim([0.0, 1.05])

    plt.xlabel('False Positive Rate')

    plt.ylabel('True Positive Rate')

    plt.title('Receiver operating characteristic example')

    plt.legend(loc="lower right")

    plt.show()
    return auc_final


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
def validate(model, validation_loader, device):
    model.eval()
    criterion = nn.BCELoss()
    pred_list = []
    label_list = []

    for inputs, targets, seq_feature in tqdm(validation_loader):
        inputs, targets, seq_feature = to_device(inputs), targets.to(device), seq_feature.to(device)
        outputs = model(inputs, seq_feature)
        valid_loss = criterion(outputs.squeeze(), targets.float())
        pred_list.extend(outputs.squeeze().cpu().detach().numpy())
        label_list.extend(targets.squeeze().cpu().detach().numpy())

    valid_score = cal_score(pred_list, label_list)
    auc_final = cal_AUC(pred_list, label_list)

    return valid_score, valid_loss, auc_final