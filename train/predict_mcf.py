from ultralytics import YOLO
import warnings
import os
import csv
import shutil
import yaml

warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

config_yaml = 'yolo11n_mcf(trimodal)'
weights_dir = 'E:/MCF-Net/runs/detect/yolo11n_mcf(trimodal)'
device = '1'
lmc_multi_modal = True
lmc_multi_modal_m3data = False

if __name__ == '__main__':
    model = YOLO(weights_dir + '/weights/best.pt')

    with open('E:/MCF-Net/ultralytics/cfg/datasets/ei-multimodal.yaml', 'r', encoding='utf-8') as f:
        data_cfg = yaml.safe_load(f)
        val_path = os.path.join(data_cfg['path'], data_cfg['val'])

    metrics = model.predict(
        source=val_path,
        imgsz=640,
        device=device,
        batch=16,
        lmc_multi_modal=lmc_multi_modal,
        lmc_multi_modal_m3data=lmc_multi_modal_m3data,
        name=weights_dir + '/eval_lmc',
        save_conf=True,
        save=True,
    )

    script_path = os.path.abspath(__file__)
    script_name = 'predict.py'
    shutil.copy(script_path, os.path.join(weights_dir, script_name))
