import mlflow, torch, os

from detectron2.engine import HookBase
from detectron2.config import CfgNode

from mlflow.entities import Metric
from mlflow.tracking import MlflowClient
from mlflow.utils.time_utils import get_current_time_millis

# define detectron2 hook for mlflow auto logging
class MLflowHook(HookBase):
    """
    A custom hook class that logs artifacts, metrics, and parameters to MLflow.
    """

    def __init__(self, cfg: CfgNode, nested_run: bool = False, upload_model: bool = False):
        super().__init__()
        self.cfg = cfg.clone()
        self.nested_run = nested_run
        self.upload_model = upload_model
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
            model_path = os.path.join(self.cfg.OUTPUT_DIR, "model_final.pth")
            if (self.upload_model and os.path.exists(model_path)):
                os.remove(model_path)

            mlflow.log_artifacts(self.cfg.OUTPUT_DIR)
            mlflow.end_run()
