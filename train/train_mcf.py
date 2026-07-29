from ultralytics import YOLO
import warnings
import os
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

config_yaml = 'yolo11n_mcf(trimodal)'
device = '1'
lmc_multi_modal = True
lmc_multi_modal_m3data = False

if __name__ == '__main__':
    model = YOLO(config_yaml + ".yaml")
    new_indices = list(range(4, 15)) + list(range(15, 26)) + list(range(26, 37)) + list(range(40, 53))
    old_indices = list(range(0, 11)) + list(range(0, 11)) + list(range(0, 11)) + list(range(11, 24))
    model.load_proj("yolo11n.pt", idx_mapping = dict(zip(new_indices, old_indices)))

    # Train the model on the COCO8 dataset for 100 epochs
    train_results = model.train(
        data= 'ei-multimodal.yaml',  # Path to dataset configuration file
        epochs=200,  # Number of training epochs
        #batch=64,
        batch=16,
        imgsz=640,  # Image size for training
        device=device,  # Device to run on (e.g., 'cpu', 0, [0,1,2,3])
        name= config_yaml,
        lmc_multi_modal = lmc_multi_modal,
        lmc_multi_modal_m3data=lmc_multi_modal_m3data,
        pretrained=True,
        close_mosaic=0,
        save_period=100,
        patience=800,
        workers=8,
    )