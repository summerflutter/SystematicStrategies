from dataclasses import dataclass
import os

@dataclass
class PipelineConfig:
    data_path: str = "/mnt/data/sample_ecn_data.csv"
    out_dir: str = "/mnt/data/ecn_pipeline_outputs"
    grid_ms: int = 100
    delta_ms: int = 100
    cls_threshold: float = 0.5
    fee_bps: float = 0.1
    slip_bps: float = 0.05

    def ensure_dirs(self):
        os.makedirs(self.out_dir, exist_ok=True)
        return self
