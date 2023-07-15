# Some basic setup:
import torch, detectron2, mlflow
from detectron2.utils import comm

TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
GPU_COUNT = torch.cuda.device_count()
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION, "; gpus: ", GPU_COUNT, "; world_size: ", comm.get_world_size())
print("detectron2:", detectron2.__version__)

# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger(output="logs/detectron2.log")

# import some common libraries
import os, shutil
print("cores: ", os.cpu_count())

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import load_coco_json, register_coco_instances

# import own modules
from util import MLflowHook, Trainer

# register to DatasetCatalog
register_coco_instances("dataset_train", {}, "dataset/train/result.json", "dataset/train/")
register_coco_instances("dataset_validate", {}, "dataset/validate/result.json", "dataset/validate/")

# load metadata
load_coco_json("dataset/train/result.json", "dataset/train/", "dataset_train")
load_coco_json("dataset/validate/result.json", "dataset/validate/", "dataset_validate")

# get metadata
roof_metadata = MetadataCatalog.get("dataset_train")

# setup mlflow enpoint
mlflow.set_tracking_uri("https://mlflow.ingrim4.me")
mlflow.set_experiment("big-data-prak-train")

# setup default config
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.DATASETS.TRAIN = ("dataset_train",)
cfg.DATASETS.TEST = ("dataset_validate",)
cfg.TEST.EVAL_PERIOD = 100
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 4  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.0000625  # pick a good LR
cfg.SOLVER.MAX_ITER = 5000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 2000   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(roof_metadata.thing_classes)  # only has one class (ballon).

shutil.rmtree(cfg.OUTPUT_DIR, ignore_errors=True)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

trainer = Trainer(cfg)
trainer.register_hooks(hooks=[MLflowHook(cfg, upload_artifacts=True)])
trainer.resume_or_load(resume=False)
trainer.train()

# validate with self
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth") 
predictor = DefaultPredictor(cfg)

evaluator_folder = os.path.join(cfg.OUTPUT_DIR, "validation-evaluation")
os.makedirs(evaluator_folder, exist_ok=True)

evaluator = COCOEvaluator("dataset_validate", output_dir=evaluator_folder)
val_loader = build_detection_test_loader(cfg, "dataset_validate")
evaluation_results = inference_on_dataset(predictor.model, val_loader, evaluator)

for k, v in evaluation_results["bbox"].items():
    mlflow.log_metric(f"validation/{k}", 100 - v, step=0)

mlflow.log_artifacts(evaluator_folder, "validation-evaluation")
mlflow.log_text(str(evaluation_results), "validation-evaluation/coco-metrics.txt")

# end run
mlflow.end_run()
