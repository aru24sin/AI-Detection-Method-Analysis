import torch
import numpy as np
import matplotlib.pyplot as plt
from networks.resnet import resnet50
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             accuracy_score, precision_score, recall_score,
                             confusion_matrix, roc_auc_score, roc_curve)
from options.test_options import TestOptions
from data import create_dataloader

def validate(model, opt):
    data_loader = create_dataloader(opt)

    with torch.no_grad():
        y_true, y_pred_probs = [], []
        for img, label in data_loader:
            in_tens = img.cuda()
            out_tens = model(in_tens).sigmoid().flatten()
            y_pred_probs.extend(out_tens.tolist())
            y_true.extend(label.flatten().tolist())

    y_true, y_pred_probs = np.array(y_true), np.array(y_pred_probs)
    y_pred_labels = y_pred_probs > 0.5  # Thresholding to get predicted labels

    r_acc = accuracy_score(y_true[y_true == 0], y_pred_labels[y_true == 0])
    f_acc = accuracy_score(y_true[y_true == 1], y_pred_labels[y_true == 1])
    acc = accuracy_score(y_true, y_pred_labels)
    ap = average_precision_score(y_true, y_pred_probs)
    precision = precision_score(y_true, y_pred_labels)
    recall = recall_score(y_true, y_pred_labels)
    roc_auc = roc_auc_score(y_true, y_pred_probs)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred_labels, normalize='all').ravel()

    # Compute ROC curve
    print("the predicted proability and labels:" + str(len(y_pred_labels)))
    #for t_cur in range(0, 5):
    #    print("true lable: " + str(y_true[t_cur]))
    #    print(y_pred_probs[t_cur])
    #    print(y_pred_labels[t_cur])
   
    fpr, tpr, roc_threshold = roc_curve(y_true, y_pred_probs, pos_label=1)
    #fpr, tpr, roc_threshold = roc_curve(y_true, y_pred_labels, pos_label=1)
    roc_data = (fpr, tpr, roc_threshold)

    return acc, ap, r_acc, f_acc, precision, recall, roc_auc, tn, fp, fn, tp, roc_data, y_true, y_pred_probs

if __name__ == '__main__':
    opt = TestOptions().parse(print_options=False)

    model = resnet50(num_classes=1)
    state_dict = torch.load(opt.model_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()

    acc, avg_precision, r_acc, f_acc, precision, recall, roc_auc, tn, fp, fn, tp, roc_data, y_true, y_pred_probs = validate(
        model, opt)

    print("accuracy:", acc)
    print("average precision:", avg_precision)
    print("accuracy of real images:", r_acc)
    print("accuracy of fake images:", f_acc)
    print("precision:", precision)
    print("recall:", recall)
    print("ROC AUC:", roc_auc)
    print("TP:", tp)
    print("FP:", fp)
    print("FN:", fn)
    print("TN:", tn)


    # Plot and save ROC curve
    fpr, tpr = roc_data
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.close()
