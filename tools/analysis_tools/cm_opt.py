#!/usr/bin/env python
import argparse
import os
import itertools
import numpy as np
from mmengine import Config, DictAction
from mmengine.fileio import load
from mmengine.registry import init_default_scope

from mmdet.registry import DATASETS

# Import functions from the confusion matrix script
from mmdet.evaluation import bbox_overlaps
from mmcv.ops import nms
from mmdet.utils import replace_cfg_vals, update_data_root


def parse_args():
    parser = argparse.ArgumentParser(
        description='Find optimal thresholds for precision and recall')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'prediction_path', help='prediction path where test .pkl result')
    parser.add_argument(
        'save_dir', help='directory where results will be saved')
    parser.add_argument(
        '--score-thrs',
        type=float,
        nargs='+',
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        help='score thresholds to evaluate')
    parser.add_argument(
        '--tp-iou-thrs',
        type=float,
        nargs='+',
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
        help='IoU thresholds to be considered as matched')
    parser.add_argument(
        '--nms-iou-thrs',
        type=float,
        nargs='+',
        default=[None,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        help='nms IoU thresholds')
    parser.add_argument(
        '--metrics',
        type=str,
        nargs='+',
        default=['precision', 'recall', 'f1', 'macro_f1', 'weighted_f1'],
        choices=['precision', 'recall', 'f1', 'macro_f1', 'weighted_f1', 'all'],
        help='metrics to optimize (precision, recall, f1, macro_f1, weighted_f1, or all)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file.')
    args = parser.parse_args()
    
    # If 'all' is specified, include all metrics
    if 'all' in args.metrics:
        args.metrics = ['precision', 'recall', 'f1', 'macro_f1', 'weighted_f1']
        
    return args


def calculate_confusion_matrix(dataset, results, score_thr=0, nms_iou_thr=None, tp_iou_thr=0.5):
    """Calculate the confusion matrix."""
    num_classes = len(dataset.metainfo['classes'])
    confusion_matrix = np.zeros(shape=[num_classes + 1, num_classes + 1])
    
    for idx, per_img_res in enumerate(results):
        res_bboxes = per_img_res['pred_instances']
        gts = dataset.get_data_info(idx)['instances']
        analyze_per_img_dets(confusion_matrix, gts, res_bboxes, score_thr,
                             tp_iou_thr, nms_iou_thr)
    
    return confusion_matrix


def analyze_per_img_dets(confusion_matrix, gts, result, score_thr=0, tp_iou_thr=0.5, nms_iou_thr=None):
    """Analyze detection results on each image."""
    true_positives = np.zeros(len(gts))
    gt_bboxes = []
    gt_labels = []
    for gt in gts:
        gt_bboxes.append(gt['bbox'])
        gt_labels.append(gt['bbox_label'])

    gt_bboxes = np.array(gt_bboxes)
    gt_labels = np.array(gt_labels)

    unique_label = np.unique(result['labels'].numpy())

    for det_label in unique_label:
        mask = (result['labels'] == det_label)
        det_bboxes = result['bboxes'][mask].numpy()
        det_scores = result['scores'][mask].numpy()

        if nms_iou_thr:
            det_bboxes, _ = nms(
                det_bboxes, det_scores, nms_iou_thr, score_threshold=score_thr)
        ious = bbox_overlaps(det_bboxes[:, :4], gt_bboxes)
        for i, score in enumerate(det_scores):
            det_match = 0
            if score >= score_thr:
                for j, gt_label in enumerate(gt_labels):
                    if ious[i, j] >= tp_iou_thr:
                        det_match += 1
                        if gt_label == det_label:
                            true_positives[j] += 1  # TP
                        confusion_matrix[gt_label, det_label] += 1
                if det_match == 0:  # BG FP
                    confusion_matrix[-1, det_label] += 1
    for num_tp, gt_label in zip(true_positives, gt_labels):
        if num_tp == 0:  # FN
            confusion_matrix[gt_label, -1] += 1


def calculate_metrics(confusion_matrix):
    """Calculate precision, recall, and F1 from confusion matrix."""
    num_classes = confusion_matrix.shape[0] - 1  # Exclude background
    
    # Initialize metrics
    class_precision = np.zeros(num_classes)
    class_recall = np.zeros(num_classes)
    class_f1 = np.zeros(num_classes)
    
    # Calculate class instances for weighted metrics
    class_instances = np.zeros(num_classes)
    for i in range(num_classes):
        # Total ground truth instances of this class
        class_instances[i] = confusion_matrix[i, :].sum()
    
    total_instances = class_instances.sum()
    class_weights = class_instances / total_instances if total_instances > 0 else np.zeros_like(class_instances)
    
    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp_sum = 0
        for j in range(num_classes):
            if j != i:
                fp_sum += confusion_matrix[j, i]  # Other classes predicted as class i
        fp_sum += confusion_matrix[-1, i]  # Background predicted as class i
        
        fn_sum = 0
        for j in range(num_classes):
            if j != i:
                fn_sum += confusion_matrix[i, j]  # Class i predicted as other classes
        fn_sum += confusion_matrix[i, -1]  # Class i predicted as background
        
        # Calculate metrics
        if tp + fp_sum > 0:
            class_precision[i] = tp / (tp + fp_sum)
        else:
            class_precision[i] = 0
            
        if tp + fn_sum > 0:
            class_recall[i] = tp / (tp + fn_sum)
        else:
            class_recall[i] = 0
            
        if class_precision[i] + class_recall[i] > 0:
            class_f1[i] = 2 * (class_precision[i] * class_recall[i]) / (class_precision[i] + class_recall[i])
        else:
            class_f1[i] = 0
    
    # Mean across all classes (macro metrics)
    macro_precision = np.mean(class_precision)
    macro_recall = np.mean(class_recall)
    macro_f1 = np.mean(class_f1)
    
    # Weighted metrics
    weighted_precision = np.sum(class_precision * class_weights)
    weighted_recall = np.sum(class_recall * class_weights)
    weighted_f1 = np.sum(class_f1 * class_weights)
    
    return {
        'precision': macro_precision,  # Same as macro_precision for compatibility
        'recall': macro_recall,        # Same as macro_recall for compatibility
        'f1': macro_f1,                # Same as macro_f1 for compatibility
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'class_precision': class_precision,
        'class_recall': class_recall,
        'class_f1': class_f1,
        'class_weights': class_weights
    }


# This function is no longer used as the logic has been moved to main()
def find_optimal_thresholds(dataset, results, score_thrs, tp_iou_thrs, nms_iou_thrs, metric='f1'):
    """Find optimal thresholds for the specified metric."""
    best_score = -1
    best_thresholds = None
    results_data = []
    
    # Total number of combinations
    total_combinations = len(score_thrs) * len(tp_iou_thrs) * len(nms_iou_thrs)
    print(f"Total combinations to evaluate: {total_combinations}")
    
    # Loop through all combinations
    counter = 0
    for score_thr, tp_iou_thr, nms_iou_thr in itertools.product(score_thrs, tp_iou_thrs, nms_iou_thrs):
        counter += 1
        print(f"Evaluating combination {counter}/{total_combinations}: score_thr={score_thr}, tp_iou_thr={tp_iou_thr}, nms_iou_thr={nms_iou_thr}")
        
        # Calculate confusion matrix
        confusion_matrix = calculate_confusion_matrix(
            dataset, results, score_thr, nms_iou_thr, tp_iou_thr)
        
        # Calculate metrics
        metrics = calculate_metrics(confusion_matrix)
        current_score = metrics[metric]
        
        # Record results
        result = {
            'score_thr': score_thr,
            'tp_iou_thr': tp_iou_thr,
            'nms_iou_thr': nms_iou_thr,
            'precision': metrics['macro_precision'],
            'recall': metrics['macro_recall'],
            'f1': metrics['macro_f1'],
            'macro_f1': metrics['macro_f1'],
            'weighted_f1': metrics['weighted_f1']
        }
        results_data.append(result)
        
        # Check if better than current best
        if current_score > best_score:
            best_score = current_score
            best_thresholds = (score_thr, tp_iou_thr, nms_iou_thr)
            print(f"New best {metric}: {best_score:.4f}, thresholds: score={score_thr}, tp_iou={tp_iou_thr}, nms_iou={nms_iou_thr}")
    
    return best_thresholds, best_score, results_data


def main():
    args = parse_args()
    
    cfg = Config.fromfile(args.config)
    
    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)
    
    # update data root according to MMDET_DATASETS
    update_data_root(cfg)
    
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    
    init_default_scope(cfg.get('default_scope', 'mmdet'))
    
    results = load(args.prediction_path)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    dataset = DATASETS.build(cfg.test_dataloader.dataset)
    
    # Dictionary to store best results for each metric
    best_results = {}
    all_results_data = []
    
    # Compute all metrics once to avoid redundant calculations
    print("Computing metrics for all threshold combinations...")
    
    # Define metric mapping for clarity
    metric_mapping = {
        'precision': 'macro_precision',
        'recall': 'macro_recall',
        'f1': 'macro_f1',
        'macro_f1': 'macro_f1',
        'weighted_f1': 'weighted_f1'
    }
    
    # Total number of combinations
    total_combinations = len(args.score_thrs) * len(args.tp_iou_thrs) * len(args.nms_iou_thrs)
    print(f"Total combinations to evaluate: {total_combinations}")
    
    # Compute all combinations and store results
    counter = 0
    for score_thr in args.score_thrs:
        for tp_iou_thr in args.tp_iou_thrs:
            for nms_iou_thr in args.nms_iou_thrs:
                counter += 1
                print(f"Evaluating combination {counter}/{total_combinations}: "
                      f"score_thr={score_thr}, tp_iou_thr={tp_iou_thr}, nms_iou_thr={nms_iou_thr}")
                
                # Calculate confusion matrix
                confusion_matrix = calculate_confusion_matrix(
                    dataset, results, score_thr, nms_iou_thr, tp_iou_thr)
                
                # Calculate metrics
                metrics = calculate_metrics(confusion_matrix)
                
                # Store results
                result = {
                    'score_thr': score_thr,
                    'tp_iou_thr': tp_iou_thr,
                    'nms_iou_thr': nms_iou_thr,
                    'precision': metrics['macro_precision'],
                    'recall': metrics['macro_recall'],
                    'f1': metrics['macro_f1'],
                    'macro_f1': metrics['macro_f1'],
                    'weighted_f1': metrics['weighted_f1']
                }
                all_results_data.append(result)
    
    # Find best for each metric
    import pandas as pd
    df = pd.DataFrame(all_results_data)
    
    # Save all results to a file
    df.to_csv(os.path.join(args.save_dir, 'all_threshold_search_results.csv'), index=False)
    
    # Find best for each specified metric
    for metric_name in args.metrics:
        actual_metric = metric_mapping.get(metric_name, metric_name)
        
        # Find best row for this metric
        best_row = df.loc[df[metric_name].idxmax()]
        best_score = best_row[metric_name]
        best_thresholds = (best_row['score_thr'], best_row['tp_iou_thr'], best_row['nms_iou_thr'])
        
        score_thr, tp_iou_thr, nms_iou_thr = best_thresholds
        print(f"\nOptimal thresholds for {metric_name}:")
        print(f"--score-thr {score_thr} --tp-iou-thr {tp_iou_thr} --nms-iou-thr {nms_iou_thr}")
        print(f"Best {metric_name}: {best_score:.4f}")
        
        # Calculate confusion matrix with best thresholds
        best_confusion_matrix = calculate_confusion_matrix(
            dataset, results, score_thr, nms_iou_thr, tp_iou_thr)
        
        # Save best metrics per class
        best_metrics = calculate_metrics(best_confusion_matrix)
        classes = dataset.metainfo['classes']
        metrics_data = []
        for i, cls_name in enumerate(classes):
            metrics_data.append({
                'class': cls_name,
                'precision': best_metrics['class_precision'][i],
                'recall': best_metrics['class_recall'][i],
                'f1': best_metrics['class_f1'][i],
                'weight': best_metrics['class_weights'][i]
            })
        
        # Create metric-specific directory
        metric_dir = os.path.join(args.save_dir, metric_name)
        if not os.path.exists(metric_dir):
            os.makedirs(metric_dir)
        
        # Save class metrics
        df_metrics = pd.DataFrame(metrics_data)
        df_metrics.to_csv(os.path.join(metric_dir, 'best_class_metrics.csv'), index=False)
        
        # Save command for confusion matrix
        with open(os.path.join(metric_dir, 'optimal_command.txt'), 'w') as f:
            cmd = f"python tools/analysis_tools/confusion_matrix.py {args.config} {args.prediction_path} {metric_dir} --score-thr {score_thr} --tp-iou-thr {tp_iou_thr}"
            if nms_iou_thr is not None:
                cmd += f" --nms-iou-thr {nms_iou_thr}"
            f.write(cmd + "\n")
            
        # Save a summary of results for this metric
        with open(os.path.join(metric_dir, 'summary.txt'), 'w') as f:
            f.write(f"Optimal thresholds for {metric_name}:\n")
            f.write(f"--score-thr {score_thr} --tp-iou-thr {tp_iou_thr} --nms-iou-thr {nms_iou_thr}\n")
            f.write(f"Best {metric_name}: {best_score:.4f}\n\n")
            
            # Add other metrics with these thresholds
            f.write("Performance with these thresholds:\n")
            for m in metric_mapping:
                f.write(f"{m}: {best_row[m]:.4f}\n")
                
    # Create a summary table of all best results
    summary_rows = []
    for metric_name in args.metrics:
        best_row = df.loc[df[metric_name].idxmax()]
        summary_rows.append({
            'metric': metric_name,
            'score_thr': best_row['score_thr'],
            'tp_iou_thr': best_row['tp_iou_thr'],
            'nms_iou_thr': best_row['nms_iou_thr'],
            'best_value': best_row[metric_name],
            'precision': best_row['precision'],
            'recall': best_row['recall'],
            'f1': best_row['f1'],
            'macro_f1': best_row['macro_f1'],
            'weighted_f1': best_row['weighted_f1']
        })
    
    # Save summary table
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(os.path.join(args.save_dir, 'optimization_summary.csv'), index=False)
    
    print("\nOptimization complete. Results saved to:", args.save_dir)


if __name__ == '__main__':
    main()