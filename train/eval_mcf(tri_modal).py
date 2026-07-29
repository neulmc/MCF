from ultralytics import YOLO
import warnings
import os
import csv
import shutil

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

config_yaml = 'yolo11n_mcf(trimodal)'
weights_dir = 'E:/MCF-Net/runs/detect/yolo11n_mcf(trimodal)'
device = '0'
lmc_multi_modal = True
lmc_multi_modal_m3data = False

if __name__ == '__main__':
    model = YOLO(weights_dir + '/weights/best.pt')
    metrics = model.val(
        data='ei-multimodal.yaml',
        imgsz=640,
        device=device,
        batch=16,
        lmc_multi_modal=lmc_multi_modal,
        lmc_multi_modal_m3data=lmc_multi_modal_m3data,
        name=weights_dir + '/eval_lmc',
        conf=0.001,
        #iou=0.7,
    )

    map50 = metrics.box.map50
    map75 = metrics.box.map75
    map = metrics.box.map
    ap50 = metrics.box.ap50
    fps = 1000 / metrics.speed['inference']
    n_l = model.n_l
    n_p = model.n_p
    n_g = model.n_g
    flops = model.flops


    with open(weights_dir + '/lmc_eval_results.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)


        writer.writerow(['mAP50', 'mAP75','mAP50-95', 'FPS', 'Layers', 'Params', 'Gradients', 'FLOPs(G)'])
        writer.writerow([
            f"{map50:.4f}",
            f"{map75:.4f}",
            f"{map:.4f}",
            f"{fps:.2f}",
            n_l,
            n_p,
            n_g,
            flops,
        ])

        writer.writerow([])
        writer.writerow(['Class', 'AP50', 'AP50-95'])

        if hasattr(metrics.box, 'ap_class_index') and metrics.box.ap50 is not None:
            for i, cls_idx in enumerate(metrics.box.ap_class_index):
                ap50_val = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0
                ap_val = metrics.box.ap[i] if hasattr(metrics.box, 'ap') and i < len(metrics.box.ap) else 0
                class_name = model.model.names.get(cls_idx, str(cls_idx))
                writer.writerow([class_name, f"{ap50_val:.4f}", f"{ap_val:.4f}"])

    script_path = os.path.abspath(__file__)
    script_name = 'eval.py'
    shutil.copy(script_path, os.path.join(weights_dir, script_name))

    print(f"✅ save results {config_yaml}_eval_results.csv")
    print(
        f"mAP50: {map50:.4f} | mAP75: {map75:.4f} | mAP50-95: {map:.4f} "
        f"| FPS: {fps:.2f} | Layers: {n_l} | Params: {n_p} | Gradients: {n_g} | FLOPs(G): {flops}")