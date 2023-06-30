import mlflow, torch, os

from detectron2.engine import HookBase, DefaultTrainer
from detectron2.evaluation import COCOEvaluator
from detectron2.config import CfgNode

from mlflow.entities import Metric, Param
from mlflow.tracking import MlflowClient
from mlflow.utils.time_utils import get_current_time_millis

# define detectron2 hook for mlflow auto logging
class MLflowHook(HookBase):
    """
    A custom hook class that logs artifacts, metrics, and parameters to MLflow.
    """

    def __init__(self, cfg: CfgNode, nested_run: bool = False, upload_artifacts: bool = False, end_run: bool = False):
        super().__init__()
        self.cfg = cfg.clone()
        self.nested_run = nested_run
        self.upload_artifacts = upload_artifacts
        self.end_run = end_run
        self.metric_batch: list[Metric] = []

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

    def collect_params(self, cfg: CfgNode, path: str = ""):
        params: list[Param] = []
        for key, value in cfg.items():
            if isinstance(value, CfgNode) or type(value) is dict:
                params.extend(self.collect_params(value, path + key + "."))
            else:
                params.append(Param(path + key, str(value)))
        return params

    def log_params(self):
        params = self.collect_params(self.cfg)
        self.client.log_batch(run_id=self.run_id, params=params)

    def before_train(self):
        with torch.no_grad():
            self.active_run = mlflow.start_run(nested=self.nested_run)
            self.client = MlflowClient()
            self.log_params()

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
            with open(os.path.join(self.cfg.OUTPUT_DIR, "config.yaml"), "w") as f:
                f.write(self.cfg.dump())

            if (self.upload_artifacts):
                mlflow.log_artifacts(self.cfg.OUTPUT_DIR)

            if (self.end_run):
                mlflow.end_run()

class Trainer(DefaultTrainer):
    """
    A custom trainer class that evaluates the model on the validation set every `_C.TEST.EVAL_PERIOD` iterations.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "test-evaluation")
            os.makedirs(output_folder, exist_ok=True)

        return COCOEvaluator(dataset_name, distributed=False, output_dir=output_folder)
