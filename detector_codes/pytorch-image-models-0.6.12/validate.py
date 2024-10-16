#!/usr/bin/env python3
""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
#!/usr/bin/env python3
""" ImageNet Validation Script """

import argparse
import os
import csv
import json
import time
import logging
import torch
import torch.nn as nn
import torch.nn.parallel
from collections import OrderedDict
from contextlib import suppress

from timm.models import create_model, apply_test_time_pool, load_checkpoint, is_model, list_models, set_fast_norm
from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_fuser, decay_batch_step, check_batch_size_retry

from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np

# Logger setup
_logger = logging.getLogger('validate')

def validate(args):
    # Initial setup
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher
    amp_autocast = suppress  # do nothing
    if args.amp:
        if has_native_amp:
            args.native_amp = True
        elif has_apex:
            args.apex_amp = True
        else:
            _logger.warning("Neither APEX or Native Torch AMP is available.")
    assert not args.apex_amp or not args.native_amp, "Only one AMP mode should be set."
    if args.native_amp:
        amp_autocast = torch.cuda.amp.autocast
        _logger.info('Validating in mixed precision with native PyTorch AMP.')
    elif args.apex_amp:
        _logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
    else:
        _logger.info('Validating in float32. AMP not enabled.')

    if args.fuser:
        set_jit_fuser(args.fuser)
    if args.fast_norm:
        set_fast_norm()

    # Create model
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=3,
        global_pool=args.gp,
        scriptable=args.torchscript)
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))

    data_config = resolve_data_config(
        vars(args),
        model=model,
        use_test_size=not args.use_train_size,
        verbose=True
    )
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config, use_test_size=not args.use_train_size)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu))).cuda()
    else:
        model = model.cuda()

    criterion = nn.CrossEntropyLoss().cuda()

    dataset = create_dataset(
        args.dataset,
        root=args.data,
        split=args.split,
        download=args.dataset_download,
        class_map=args.class_map
    )

    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=data_config['crop_pct'],
        pin_memory=args.pin_mem,
    )

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    model.eval()

    end = time.time()

    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            input = input.cuda()
            target = target.cuda()
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
            if isinstance(output, (tuple, list)):
                output = output[0]

            loss = criterion(output, target)

            # Measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # Store predictions and labels
            all_labels.extend(target.cpu().numpy())
            all_preds.extend(output.argmax(dim=1).cpu().numpy())

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            if batch_idx % args.log_freq == 0:
                _logger.info(
                    f'Test: [{batch_idx}/{len(loader)}] '
                    f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                    f'Loss {losses.val:.4f} ({losses.avg:.4f})  '
                    f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})  '
                    f'Acc@5 {top5.val:.3f} ({top5.avg:.3f})')

    # Compute additional metrics
    conf_matrix = confusion_matrix(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro')
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    roc_auc = roc_auc_score(all_labels, all_preds, multi_class='ovr')

    # Compute TP, FP, FN, TN for each class
    tp = conf_matrix.diagonal()
    fp = conf_matrix.sum(axis=0) - tp
    fn = conf_matrix.sum(axis=1) - tp
    tn = conf_matrix.sum() - (fp + fn + tp)

    # Directory setup
    model_dir = "/media/aryansingh/58A7-AC9A/results/pytorch-image-models-0.6.12/pytorch_glide"
    os.makedirs(model_dir, exist_ok=True)

    # Save ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_preds, pos_label=1)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    roc_curve_path = os.path.join(model_dir, f'roc_curve_{args.split}.png')
    plt.savefig(roc_curve_path)
    plt.close()

    # Save confusion matrix
    np.savetxt(os.path.join(model_dir, 'confusion_matrix.csv'), conf_matrix, delimiter=',')

    # Save precision, recall, roc_auc, TP, FP, FN, TN
    metrics = {
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'tp': tp.tolist(),
        'fp': fp.tolist(),
        'fn': fn.tolist(),
        'tn': tn.tolist(),
    }
    with open(os.path.join(model_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f)

    _logger.info(f' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}')
    _logger.info(f' * Precision: {precision:.3f}')
    _logger.info(f' * Recall: {recall:.3f}')
    _logger.info(f' * ROC AUC: {roc_auc:.3f}')
    _logger.info(f' * Confusion Matrix saved to: {os.path.join(model_dir, "confusion_matrix.csv")}')
    _logger.info(f' * ROC Curve saved to: {roc_curve_path}')
    _logger.info(f' * Metrics saved to: {os.path.join(model_dir, "metrics.json")}')
    _logger.info(f' * TP: {tp}')
    _logger.info(f' * FP: {fp}')
    _logger.info(f' * FN: {fn}')
    _logger.info(f' * TN: {tn}')

    return OrderedDict([
        ('loss', losses.avg),
        ('top1', top1.avg),
        ('top5', top5.avg),
        ('precision', precision),
        ('recall', recall),
        ('roc_auc', roc_auc),
        ('conf_matrix', conf_matrix.tolist()),
        ('tp', tp.tolist()),
        ('fp', fp.tolist()),
        ('fn', fn.tolist()),
        ('tn', tn.tolist()),
    ])

def main():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
    parser.add_argument('data', metavar='DIR',
                        help='path to dataset')
    parser.add_argument('--dataset', '-d', metavar='NAME', default='',
                        help='dataset type (default: ImageFolder/ImageTar if empty)')
    parser.add_argument('--split', metavar='NAME', default='validation',
                        help='dataset split (default: validation)')
    parser.add_argument('--dataset-download', action='store_true', default=False,
                        help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
    parser.add_argument('--model', '-m', metavar='NAME', default='resnet50',
                        help='model architecture (default: resnet50)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--img-size', default=None, type=int,
                        metavar='N', help='Input image dimension, uses model default if empty')
    parser.add_argument('--input-size', default=None, nargs=3, type=int,
                        metavar='N N N',
                        help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
    parser.add_argument('--use-train-size', action='store_true', default=False,
                        help='force use of train input size, even when test size is specified in pretrained cfg')
    parser.add_argument('--crop-pct', default=None, type=float,
                        metavar='N', help='Input image center crop pct')
    parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                        help='Override mean pixel value of dataset')
    parser.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                        help='Override std deviation of of dataset')
    parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                        help='Image resize interpolation type (overrides model)')
    parser.add_argument('--num-classes', type=int, default=None,
                        help='Number classes in dataset')
    parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                        help='path to class to idx mapping file (default: "")')
    parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                        help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
    parser.add_argument('--log-freq', default=10, type=int,
                        metavar='N', help='batch logging frequency (default: 10)')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('--num-gpu', type=int, default=1,
                        help='Number of GPUS to use')
    parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                        help='enable test time pool')
    parser.add_argument('--no-prefetcher', action='store_true', default=False,
                        help='disable fast prefetcher')
    parser.add_argument('--pin-mem', action='store_true', default=False,
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--channels-last', action='store_true', default=False,
                        help='Use channels_last memory layout')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='Use AMP mixed precision. Defaults to Apex, fallback to native Torch AMP.')
    parser.add_argument('--apex-amp', action='store_true', default=False,
                        help='Use NVIDIA Apex AMP mixed precision')
    parser.add_argument('--native-amp', action='store_true', default=False,
                        help='Use Native Torch AMP mixed precision')
    parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                        help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
    parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                        help='use ema version of weights if present')
    scripting_group = parser.add_argument_group()
    scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
                                 help='torch.jit.script the full model')
    scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
                                 help="Enable AOT Autograd support. (It's recommended to use this option with --fuser nvfuser together)")
    parser.add_argument('--fuser', default='', type=str,
                        help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
    parser.add_argument('--fast-norm', default=False, action='store_true',
                        help='enable experimental fast-norm')
    parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                        help='Output csv file for validation results (summary)')
    parser.add_argument('--real-labels', default='', type=str, metavar='FILENAME',
                        help='Real labels JSON file for imagenet evaluation')
    parser.add_argument('--valid-labels', default='', type=str, metavar='FILENAME',
                        help='Valid label indices txt file for validation of partial label space')
    parser.add_argument('--retry', default=False, action='store_true',
                        help='Enable batch size decay & retry for single model validation')
    args = parser.parse_args()

    setup_default_logging()

    validate(args)



if __name__ == '__main__':
    main()
