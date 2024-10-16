import os
import csv
import torch
import matplotlib.pyplot as plt
from validate import validate
from networks.resnet import resnet50
from options.test_options import TestOptions
from eval_config import *

# Running tests
opt = TestOptions().parse(print_options=False)
model_name = os.path.basename(model_path).replace('.pth', '')
rows = [["{} model testing on...".format(model_name)],
        ['testset', 'accuracy', 'avg precision', 'precision', 'recall', 'ROC AUC', 'TP', 'FP', 'FN', 'TN']]

print("{} model testing on...".format(model_name))
for v_id, val in enumerate(vals):
    opt.dataroot = os.path.join(dataroot, val)

    opt.classes = os.listdir(opt.dataroot) if multiclass[v_id] else ['']
    opt.no_resize = True  # testing without resizing by default

    model = resnet50(num_classes=1)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict['model'])
    model.cuda()
    model.eval()
    
    print("start validate")
    acc, ap, r_acc, f_acc, precision, recall, roc_auc, tn, fp, fn, tp, roc_data, _, _ = validate(model, opt)
    rows.append([val, acc, ap, precision, recall, roc_auc, tp, fp, fn, tn])
    print("({}) acc: {}; ap: {}; precision: {}; recall: {}; ROC AUC: {}; TP: {}; FP: {}; FN: {}; TN: {}".format(val, acc, ap, precision, recall, roc_auc, tp, fp, fn, tn))

    # Save ROC curve
    fpr, tpr, roc_threshold = roc_data
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic - {val}')
    plt.legend(loc="lower right")
    roc_image_path = os.path.join(results_dir, f"roc_curve_{val}.png")
    plt.savefig(roc_image_path)
    plt.close()

csv_name = os.path.join(results_dir, f"{model_name}.csv")
os.makedirs(results_dir, exist_ok=True)
with open(csv_name, 'a', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',')
    csv_writer.writerows(rows)
