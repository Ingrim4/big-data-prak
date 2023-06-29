# Some basic setup:
import torch, detectron2, mlflow, hyperopt
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
GPU_COUNT = torch.cuda.device_count()
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION, "; gpus: ", GPU_COUNT)
print("detectron2:", detectron2.__version__)

# Setup detectron2 logger
from detectron2.utils.logger import setup_logger
setup_logger(output="logs/detectron2.log")

# import some common libraries
import os, shutil, pickle, atexit
print("cores: ", os.cpu_count())

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.config import get_cfg, CfgNode
from detectron2.data import MetadataCatalog, DatasetCatalog, build_detection_test_loader
from detectron2.data.datasets import load_coco_json, register_coco_instances

# import some common mlflow utilities
from mlflow.entities import Run
from mlflow.tracking import MlflowClient

# import own modules
from util.mlflow_hook import MLflowHook

# register to DatasetCatalog
register_coco_instances("dataset_train", {}, "dataset/train/result.json", "dataset/train/")
register_coco_instances("dataset_validate", {}, "dataset/validate/result.json", "dataset/validate/")

# load metadata
load_coco_json("dataset/train/result.json", "dataset/train/", "dataset_train")
load_coco_json("dataset/validate/result.json", "dataset/validate/", "dataset_validate")

# get metadata
roof_metadata = MetadataCatalog.get("dataset_train")

# setup default config
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.DATASETS.TRAIN = ("dataset_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 2000    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 2048   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(roof_metadata.thing_classes)  # only has one class (ballon).
cfg.freeze()

def train(parameters: dict[str, any], cfg: CfgNode):
    # update "optimized" parameters
    cfg = cfg.clone()
    cfg.defrost()
    cfg.SOLVER.IMS_PER_BATCH = parameters['IMS_PER_BATCH']
    cfg.SOLVER.BASE_LR = parameters['BASE_LR']
    cfg.SOLVER.MAX_ITER = parameters['MAX_ITER']
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = parameters['BATCH_SIZE_PER_IMAGE']
    cfg.freeze()

    # clean output directory
    shutil.rmtree(cfg.OUTPUT_DIR, ignore_errors=True)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    # train model
    trainer = DefaultTrainer(cfg)
    trainer.register_hooks(hooks=[MLflowHook(cfg, nested_run=True)])
    trainer.resume_or_load(resume=False)
    trainer.train()

    # return train metrics
    return trainer.storage.latest_with_smoothing_hint(window_size=10)

def build_train_objective(metric: str):
    def train_func(parameters: dict[str, any]):
        metrics = train(parameters, cfg)
        return { 'status': hyperopt.STATUS_OK, 'loss': metrics[metric][0] }
    return train_func

def log_best(run: Run, metric: str):
    # query all child runs
    run_query = "tags.mlflow.parentRunId = '{run_id}' ".format(run_id=run.info.run_id)
    runs = MlflowClient().search_runs([run.info.experiment_id], run_query)

    # find best run
    best_run = min(runs, key=lambda run: run.data.metrics[metric])

    # log best run
    mlflow.set_tag("best_run", best_run.info.run_id)
    mlflow.log_metric(f"best_{metric}", best_run.data.metrics[metric])

# Number of hyperopt evaluations
MAX_EVALS = 32
# Metric to optimize
METRIC = "total_loss"
# Number of experiments to run at once
PARALLELISM = 1

space = {
    #'IMS_PER_BATCH': hyperopt.hp.uniformint('IMS_PER_BATCH', 1, 2),
    'IMS_PER_BATCH': hyperopt.hp.uniformint('IMS_PER_BATCH', 1, 4),
    'BASE_LR': hyperopt.hp.uniform('BASE_LR', 1e-5, 1e-3),
    #'MAX_ITER': hyperopt.hp.uniformint('MAX_ITER', 100, 1000),
    'MAX_ITER': hyperopt.hp.uniformint('MAX_ITER', 1024, 4096),
    #'BATCH_SIZE_PER_IMAGE': hyperopt.hp.uniformint('BATCH_SIZE_PER_IMAGE', 64, 1024)
    'BATCH_SIZE_PER_IMAGE': hyperopt.hp.uniformint('BATCH_SIZE_PER_IMAGE', 512, 3000)
}

trials = hyperopt.Trials()

def save_trials():
    pickle.dump(trials, open("trials.p", "wb"))
atexit.register(save_trials)

with mlflow.start_run() as run:
    best = hyperopt.fmin(
        fn=build_train_objective(METRIC),
        space=space,
        algo=hyperopt.tpe.suggest, # alternative hyperopt.rand.suggest
        max_evals=MAX_EVALS,
        trials=trials
    )
    mlflow.set_tag("best_params", str(best))
    log_best(run, METRIC)