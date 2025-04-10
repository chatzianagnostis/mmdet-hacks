#!/usr/bin/env python
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator
from mmcv.ops import nms
from mmengine import Config, DictAction
from mmengine.fileio import load
from mmengine.registry import init_default_scope
from mmengine.utils import ProgressBar

from mmdet.evaluation import bbox_overlaps
from mmdet.registry import DATASETS
from mmdet.utils import replace_cfg_vals, update_data_root


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate confusion matrix from detection results')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        'prediction_path', help='prediction path where test .pkl result')
    parser.add_argument(
        'save_dir', help='directory where confusion matrix will be saved')
    parser.add_argument(
        '--show', action='store_true', help='show confusion matrix')
    parser.add_argument(
        '--color-theme',
        default='plasma',
        help='theme of the matrix color map')
    parser.add_argument(
        '--score-thr',
        type=float,
        default=0.3,
        help='score threshold to filter detection bboxes')
    parser.add_argument(
        '--tp-iou-thr',
        type=float,
        default=0.5,
        help='IoU threshold to be considered as matched')
    parser.add_argument(
        '--nms-iou-thr',
        type=float,
        default=None,
        help='nms IoU threshold, only applied when users want to change the'
        'nms IoU threshold.')
    parser.add_argument(
        '--normalize',
        action='store_true',
        help='normalize the confusion matrix')
    parser.add_argument(
        '--save-matrix',
        action='store_true',
        help='save the confusion matrix as numpy file')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()
    return args


def calculate_confusion_matrix(dataset,
                               results,
                               score_thr=0,
                               nms_iou_thr=None,
                               tp_iou_thr=0.5):
    """Calculate the confusion matrix.

    Args:
        dataset (Dataset): Test or val dataset.
        results (list[ndarray]): A list of detection results in each image.
        score_thr (float|optional): Score threshold to filter bboxes.
            Default: 0.
        nms_iou_thr (float|optional): nms IoU threshold, the detection results
            have done nms in the detector, only applied when users want to
            change the nms IoU threshold. Default: None.
        tp_iou_thr (float|optional): IoU threshold to be considered as matched.
            Default: 0.5.
    """
    num_classes = len(dataset.metainfo['classes'])
    confusion_matrix = np.zeros(shape=[num_classes + 1, num_classes + 1])
    
    assert len(dataset) == len(results)
    prog_bar = ProgressBar(len(results))
    
    # Track total ground truths
    total_gt_count = 0
    
    for idx, per_img_res in enumerate(results):
        res_bboxes = per_img_res['pred_instances']
        gts = dataset.get_data_info(idx)['instances']
        total_gt_count += len(gts)
        
        analyze_per_img_dets(confusion_matrix, gts, res_bboxes, score_thr,
                             tp_iou_thr, nms_iou_thr)
        prog_bar.update()
    
    # Verify ground truth count matches what's in the confusion matrix
    matrix_gt_count = 0
    for i in range(num_classes):
        matrix_gt_count += confusion_matrix[i].sum()  # Sum across the row (all predictions for this gt class)
    
    print(f"Total ground truth objects: {total_gt_count}")
    print(f"Total ground truth in confusion matrix: {matrix_gt_count}")
    
    return confusion_matrix


def analyze_per_img_dets(confusion_matrix,
                         gts,
                         result,
                         score_thr=0,
                         tp_iou_thr=0.5,
                         nms_iou_thr=None):
    """Analyze detection results on each image.

    Args:
        confusion_matrix (ndarray): The confusion matrix,
            has shape (num_classes + 1, num_classes + 1).
        gts (list): Ground truth instances.
        result (dict): Detection results.
        score_thr (float): Score threshold to filter bboxes.
            Default: 0.
        tp_iou_thr (float): IoU threshold to be considered as matched.
            Default: 0.5.
        nms_iou_thr (float|optional): nms IoU threshold, the detection results
            have done nms in the detector, only applied when users want to
            change the nms IoU threshold. Default: None.
    """
    # Extract ground truth bboxes and labels
    gt_bboxes = []
    gt_labels = []
    for gt in gts:
        gt_bboxes.append(gt['bbox'])
        gt_labels.append(gt['bbox_label'])

    # Handle empty ground truth case
    if not gt_bboxes:
        gt_bboxes = np.zeros((0, 4))
        gt_labels = np.zeros(0, dtype=np.int64)
    else:
        gt_bboxes = np.array(gt_bboxes)
        gt_labels = np.array(gt_labels)
    
    # Track which ground truths have been matched
    gt_matched = np.zeros(len(gt_labels), dtype=bool)
    
    # Handle detection processing
    if 'labels' in result and len(result['labels']) > 0:
        # Extract unique detection labels
        unique_labels = np.unique(result['labels'].numpy())
        
        # Dictionary to store filtered detections by class
        filtered_detections = {}
        
        # For each detection class
        for det_label in unique_labels:
            # Get detections of this class
            mask = (result['labels'] == det_label)
            det_bboxes = result['bboxes'][mask].numpy()
            det_scores = result['scores'][mask].numpy()
            
            # Apply additional NMS if specified
            if nms_iou_thr is not None and len(det_bboxes) > 0:
                det_bboxes_with_scores = np.hstack([det_bboxes, det_scores[:, None]])
                keep_inds = nms(det_bboxes_with_scores, nms_iou_thr)[0]
                if keep_inds is not None and len(keep_inds) > 0:
                    det_bboxes = det_bboxes[keep_inds]
                    det_scores = det_scores[keep_inds]
            
            # Filter by score threshold
            if len(det_bboxes) > 0:
                score_mask = det_scores >= score_thr
                det_bboxes = det_bboxes[score_mask]
                det_scores = det_scores[score_mask]
            
            # Store filtered detections
            filtered_detections[det_label] = {
                'bboxes': det_bboxes,
                'scores': det_scores
            }
        
        # If we have ground truths
        if len(gt_bboxes) > 0:
            # For each ground truth, find best matching detection
            for j, gt_label in enumerate(gt_labels):
                best_iou = 0
                best_det_label = None
                best_det_idx = -1
                
                # Look through all detection classes
                for det_label, dets in filtered_detections.items():
                    if len(dets['bboxes']) == 0:
                        continue
                    
                    # Calculate IoU between this ground truth and all detections of this class
                    ious = bbox_overlaps(dets['bboxes'], gt_bboxes[j:j+1])
                    
                    # Find max IoU for this detection class
                    max_iou_idx = np.argmax(ious)
                    max_iou = ious[max_iou_idx][0]
                    
                    # If better than current best, update
                    if max_iou > best_iou and max_iou >= tp_iou_thr:
                        best_iou = max_iou
                        best_det_label = det_label
                        best_det_idx = max_iou_idx
                
                # If we found a matching detection
                if best_det_label is not None:
                    # Update confusion matrix - ground truth matched with detection
                    confusion_matrix[gt_label, best_det_label] += 1
                    gt_matched[j] = True
                    
                    # Remove the matched detection to prevent matching it to multiple ground truths
                    dets = filtered_detections[best_det_label]
                    dets['bboxes'] = np.delete(dets['bboxes'], best_det_idx, axis=0)
                    dets['scores'] = np.delete(dets['scores'], best_det_idx)
                else:
                    # No matching detection - false negative
                    confusion_matrix[gt_label, -1] += 1
            
            # Count remaining detections as false positives
            for det_label, dets in filtered_detections.items():
                # All remaining detections of this class are false positives
                confusion_matrix[-1, det_label] += len(dets['bboxes'])
        else:
            # No ground truths - all detections are false positives
            for det_label, dets in filtered_detections.items():
                confusion_matrix[-1, det_label] += len(dets['bboxes'])


def plot_confusion_matrix(confusion_matrix,
                          labels,
                          save_dir=None,
                          show=True,
                          title='Confusion Matrix',
                          color_theme='plasma',
                          normalize=False):
    """Draw confusion matrix with matplotlib.

    Args:
        confusion_matrix (ndarray): The confusion matrix.
        labels (list[str]): List of class names.
        save_dir (str|optional): If set, save the confusion matrix plot to the
            given path. Default: None.
        show (bool): Whether to show the plot. Default: True.
        title (str): Title of the plot. Default: `Confusion Matrix`.
        color_theme (str): Theme of the matrix color map. Default: `plasma`.
        normalize (bool): Whether to normalize the confusion matrix.
            Default: False.
    """
    # Optionally normalize the confusion matrix
    display_matrix = confusion_matrix.copy()
    if normalize:
        per_label_sums = confusion_matrix.sum(axis=1)[:, np.newaxis]
        # Avoid division by zero
        per_label_sums[per_label_sums == 0] = 1
        display_matrix = display_matrix.astype(np.float32) / per_label_sums * 100
        title = 'Normalized ' + title

    num_classes = len(labels)
    fig, ax = plt.subplots(
        figsize=(0.5 * num_classes, 0.5 * num_classes * 0.8), dpi=180)
    cmap = plt.get_cmap(color_theme)
    im = ax.imshow(display_matrix, cmap=cmap)
    plt.colorbar(mappable=im, ax=ax)

    title_font = {'weight': 'bold', 'size': 12}
    ax.set_title(title, fontdict=title_font)
    label_font = {'size': 10}
    plt.ylabel('Ground Truth Label', fontdict=label_font)
    plt.xlabel('Prediction Label', fontdict=label_font)

    # draw locator
    xmajor_locator = MultipleLocator(1)
    xminor_locator = MultipleLocator(0.5)
    ax.xaxis.set_major_locator(xmajor_locator)
    ax.xaxis.set_minor_locator(xminor_locator)
    ymajor_locator = MultipleLocator(1)
    yminor_locator = MultipleLocator(0.5)
    ax.yaxis.set_major_locator(ymajor_locator)
    ax.yaxis.set_minor_locator(yminor_locator)

    # draw grid
    ax.grid(True, which='minor', linestyle='-')

    # draw label
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.tick_params(
        axis='x', bottom=False, top=True, labelbottom=False, labeltop=True)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha='left', rotation_mode='anchor')

    # draw confusion matrix value
    for i in range(num_classes):
        for j in range(num_classes):
            value = display_matrix[i, j]
            if normalize:
                ax.text(
                    j,
                    i,
                    f'{value:.1f}%' if not np.isnan(value) else '-',
                    ha='center',
                    va='center',
                    color='w',
                    size=7)
            else:
                ax.text(
                    j,
                    i,
                    f'{int(value)}' if not np.isnan(value) else '-',
                    ha='center',
                    va='center',
                    color='w',
                    size=7)

    ax.set_ylim(len(display_matrix) - 0.5, -0.5)  # matplotlib>3.1.1

    fig.tight_layout()
    if save_dir is not None:
        plt.savefig(
            os.path.join(save_dir, 'confusion_matrix.png'), format='png')
    if show:
        plt.show()


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

    confusion_matrix = calculate_confusion_matrix(dataset, results,
                                                  args.score_thr,
                                                  args.nms_iou_thr,
                                                  args.tp_iou_thr)
    
    # Calculate metrics from confusion matrix
    num_classes = len(dataset.metainfo['classes'])
    class_names = dataset.metainfo['classes']
    metrics = {}
    
    for i in range(num_classes):
        # Extract values from confusion matrix
        tp = confusion_matrix[i, i]
        fn = confusion_matrix[i, -1]
        fp = confusion_matrix[-1, i]
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[class_names[i]] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # Calculate macro averages
    macro_precision = np.mean([metrics[cls]['precision'] for cls in class_names])
    macro_recall = np.mean([metrics[cls]['recall'] for cls in class_names])
    macro_f1 = np.mean([metrics[cls]['f1'] for cls in class_names])
    
    # Save metrics to file
    with open(os.path.join(args.save_dir, 'metrics.txt'), 'w') as f:
        f.write(f"Confusion Matrix Metrics (score_thr={args.score_thr}, tp_iou_thr={args.tp_iou_thr})\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Per-class metrics:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1':<12}\n")
        f.write("-" * 60 + "\n")
        
        for cls in class_names:
            f.write(f"{cls:<20} {metrics[cls]['precision']:<12.4f} {metrics[cls]['recall']:<12.4f} {metrics[cls]['f1']:<12.4f}\n")
        
        f.write("\nMacro averages:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Macro':<20} {macro_precision:<12.4f} {macro_recall:<12.4f} {macro_f1:<12.4f}\n")
    
    # Save the raw confusion matrix if requested
    if args.save_matrix:
        np.save(os.path.join(args.save_dir, 'confusion_matrix.npy'), confusion_matrix)
    
    # Plot the confusion matrix
    plot_confusion_matrix(
        confusion_matrix,
        dataset.metainfo['classes'] + ('background', ),
        save_dir=args.save_dir,
        show=args.show,
        color_theme=args.color_theme,
        normalize=args.normalize)


if __name__ == '__main__':
    main()