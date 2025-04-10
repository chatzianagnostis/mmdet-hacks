import os
import pickle
import time
import cv2
import torch
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector
from mmdet.utils import register_all_modules
from mmdet.visualization import DetLocalVisualizer

CLASS_NAMES = [
    "blister", "dent", "orange", "scratch", "stuck", "other", "paintless"
]

def process_folder(
    config_file,
    checkpoint_file,
    input_folder,
    output_folder,
    score_threshold=0.01,
    device='cuda:0',
    save_pickle_path='predictions.pkl'
):
    register_all_modules()
    model = init_detector(config_file, checkpoint_file, device=device)

    visualizer = DetLocalVisualizer()
    visualizer.dataset_meta = {"classes": CLASS_NAMES}

    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]

    all_results = []
    total_infer_time = 0.0

    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(input_folder, image_file)
        image = cv2.imread(image_path)

        # Inference
        torch.cuda.synchronize()
        start = time.perf_counter()
        result = inference_detector(model, image)
        torch.cuda.synchronize()
        end = time.perf_counter()
        total_infer_time += (end - start)

        # Attach metainfo
        result.set_metainfo({'img_path': image_path})

        # Move tensors to CPU for compatibility
        pred = result.pred_instances
        pred = pred.cpu()  # moves all tensors inside to CPU

        result_dict = {
            'pred_instances': pred,
            'metainfo': result.metainfo
        }
        all_results.append(result_dict)

        # Filtered predictions for visualization/text
        pred_instances = result.pred_instances
        mask = pred_instances.scores > score_threshold

        bboxes = pred_instances.bboxes[mask].cpu().numpy()
        scores = pred_instances.scores[mask].cpu().numpy()
        labels = pred_instances.labels[mask].cpu().numpy()
        label_names = [CLASS_NAMES[label] for label in labels]

        # Save visualization
        vis_path = os.path.join(output_folder, f'pred_{image_file}')
        visualizer.add_datasample(
            name='',
            image=image,
            data_sample=result,
            draw_gt=False,
            pred_score_thr=score_threshold,
            out_file=vis_path
        )

        # Save detections to .txt
        txt_path = os.path.join(output_folder, f'pred_{os.path.splitext(image_file)[0]}.txt')
        with open(txt_path, 'w') as f:
            for i in range(len(bboxes)):
                x1, y1, x2, y2 = bboxes[i]
                score = scores[i]
                class_name = label_names[i]
                f.write(f'{class_name} {score:.3f} {x1:.1f} {y1:.1f} {x2:.1f} {y2:.1f}\n')

    # Save .pkl in MMDet-compatible format
    pkl_path = os.path.join(output_folder, save_pickle_path)
    with open(pkl_path, 'wb') as f:
        pickle.dump(all_results, f)

    print(f"\n✅ Saved predictions for confusion matrix: {pkl_path}")
    print(f"⏱️  Avg inference time/image: {total_infer_time / len(image_files):.6f} seconds")



if __name__ == '__main__':
    config_file = 'd:\\vagg\\mmdetection\\work_dirs\\rtmdet-ins_tiny_8xb32-300e_coco1024\\rtmdet-ins_tiny_8xb32-300e_coco.py'
    checkpoint_file = 'd:\\vagg\\mmdetection\\work_dirs\\rtmdet-ins_tiny_8xb32-300e_coco1024\\best_coco_bbox_mAP_50_epoch_492.pth'
    input_folder = 'D:\\vagg\\mmdetection\\data\\test\\images'
    output_folder = 'D:\\vagg\\mmdetection\\DS-PR-004_Production_profiles_bench\\test\\img'

    process_folder(
        config_file=config_file,
        checkpoint_file=checkpoint_file,
        input_folder=input_folder,
        output_folder=output_folder,
        score_threshold=0.01,
        save_pickle_path='predictions.pkl'
    )
