import random
import time





def random_evaluation(**kwargs):
    accuracy = round(random.uniform(0.5, 1.0),1)

    time.sleep(1)  # Simulate a time-consuming evaluation
    return {**kwargs, "accuracy": accuracy}



import multiViewLearning

def train_model(**kwargs):
    from pathlib import Path
import torch

from multiViewLearning.evaluationfn import run_experiment
from multiViewLearning.utils import get_device

config = {
    "exp_name": "tomate_rgb_lbp_smoke",
    "root": str(Path("views").resolve()),
    "views": ["RGB", "LBP"],
    "modes": ["RGB", "L"],
    "datasetType": "multiview",
    "backbone": "resnet18",
    "gated": True,
    "pretrained": True,
    "image_size": 224,
    "batch_size": 4,
    "epochs": 1,
    "lr": 0.001,
    "lr_step": 5,
    "lr_gamma": 0.5,
    "train_ratio": 0.8,
    "use_sampler": False,
    "out_dir": "results/tomate_rgb_lbp_smoke",
}

device = get_device()
res = run_experiment(config, device)
print("Model saved to:", res.model_path)

def parse(**kwargs):
    parsed = {}
    if "views" in kwargs:
        parsed["views"] = str(kwargs["views"])