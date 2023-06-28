# Some basic setup:
import torch, detectron2, mlflow, hyperopt
from detectron2.utils import comm

# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger(output="logs/detectron2.log")

# import some common libraries
import os, shutil, pickle, atexit
print("cores: ", os.cpu_count())

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import HookBase, DefaultTrainer, DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg, CfgNode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import load_coco_json, register_coco_instances

# import some common mlflow utilities
from mlflow.entities import Metric, Run
from mlflow.tracking import MlflowClient
from mlflow.utils.time_utils import get_current_time_millis

# define detectron2 hook for mlflow auto logging
class MLflowHook(HookBase):
    """
    A custom hook class that logs artifacts, metrics, and parameters to MLflow.
    """

    def __init__(self, cfg: CfgNode, nested_run: bool):
        super().__init__()
        self.cfg = cfg.clone()
        self.nested_run = nested_run
        self.metric_batch = []

    @property
    def run_id(self):
        return self.active_run.info.run_id

    def flush_metric_batch(self):
        self.client.log_batch(run_id=self.run_id, metrics=self.metric_batch)
        self.metric_batch.clear()

    def log_metric(self, entry: Metric):
        self.metric_batch.append(entry)
        if len(self.metric_batch) > 512:
            self.flush_metric_batch()

    def before_train(self):
        with torch.no_grad():
            self.active_run = mlflow.start_run(nested=self.nested_run)
            self.client = MlflowClient()

            mlflow.log_params({ ('SOLVER.'+k): v for k, v in self.cfg.SOLVER.items() })
            mlflow.log_params({ ('MODEL.ROI_HEADS.'+k): v for k, v in self.cfg.MODEL.ROI_HEADS.items() })

    def after_step(self):
        with torch.no_grad():
            latest_metrics = self.trainer.storage.latest()
            for k, v in latest_metrics.items():
                self.log_metric(Metric(
                    key=k,
                    value=v[0],
                    step=v[1],
                    timestamp=get_current_time_millis()
                ))

    def after_train(self):
        with torch.no_grad():
            if len(self.metric_batch) > 0:
                self.flush_metric_batch()
            with open(os.path.join(self.cfg.OUTPUT_DIR, "model-config.yaml"), "w") as f:
                f.write(self.cfg.dump())

            # delete model file cause my server is tiny
            model_path = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
            if (os.path.exists(model_path)):
                os.remove(model_path)

            mlflow.log_artifacts(self.cfg.OUTPUT_DIR)
            mlflow.end_run()

class CocoTrainer(DefaultTrainer):
    """
    A custom trainer class that evaluates the model on the validation set every `_C.TEST.EVAL_PERIOD` iterations.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs(cfg.OUTPUT_DIR_VALIDATION_SET_EVALUATION,
                        exist_ok=True)

        return COCOEvaluator(dataset_name, distributed=False, output_dir=cfg.OUTPUT_DIR_VALIDATION_SET_EVALUATION)

# register to DatasetCatalog
register_coco_instances("train_dataset", {}, "train_dataset/result.json", "train_dataset/")

# load dataset and metadata
dataset_dicts = load_coco_json("train_dataset/result.json", "train_dataset/", "train_dataset")
roof_metadata = MetadataCatalog.get("train_dataset")

# setup mlflow enpoint
mlflow.set_tracking_uri("https://mlflow.ingrim4.me")
mlflow.set_experiment("big-data-prak-maurice")

# setup default config
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.DATASETS.TRAIN = ("train_dataset",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.REFERENCE_WORLD_SIZE = 2
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.000125  # pick a good LR
cfg.SOLVER.MAX_ITER = 3000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 4096   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(roof_metadata.thing_classes)  # only has one class (ballon).

shutil.rmtree(cfg.OUTPUT_DIR)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

def main():
    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    GPU_COUNT = torch.cuda.device_count()
    print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION, "; gpus: ", GPU_COUNT, "; world_size: ", comm.get_world_size())
    print("detectron2:", detectron2.__version__)

    trainer = CocoTrainer(cfg)
    trainer.register_hooks(hooks=[MLflowHook(cfg, nested_run=False)])
    trainer.resume_or_load(resume=False)
    trainer.train()

# validate with self
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

evaluator = COCOEvaluator("train_dataset", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "train_dataset")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
