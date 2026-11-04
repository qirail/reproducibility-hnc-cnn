"""
Run saved model weights (the original authors' weights, the models retrained for
external validation, or the 5-fold cross-validation fold models) and store everything in files.

Outputs (in RESULTS_DIR):
  summary.txt                      one readable block per run and dataset (AUC, CI, counts)
  predictions_<run>_<set>.csv      patient id, label, predicted probability
  run_info.txt                     python/torch versions
  raw_output.txt                   everything the code would print to the terminal

Run from the repository root:  python predict_original_weights.py
"""

import contextlib
import os
import platform
import sys
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader

from hn_cnn.cnn import MODELS
from hn_cnn.constants import *
from hn_cnn.data_augmentation import TRANSFORM_TO_TENSOR
from hn_cnn.fit import predict
from hn_cnn.parse_data import ImageDataset

################################## Settings ##################################

RESULTS_DIR = "results/statistical_reproducibility_weights/5_fold_cv"
WEIGHTS_DIR = "results/statistical_reproducibility/seeds/seed_11/models"          # folder that holds the downloaded .pth.tar file | (with_clinical_data models can be in the same folder)

CANADA = {CLINICAL_DATA_PATH: "data/pre-processed/canada.csv", SCANS_PATH: "data/pre-processed/canada_c"}
MAASTRO = {CLINICAL_DATA_PATH: "data/pre-processed/maastro.csv", SCANS_PATH: "data/pre-processed/maastro_c"}
# External cohort (LRF and DM). Adjust the two names to the csv/png folder you actually use.
CMC = {CLINICAL_DATA_PATH: "data/pre-processed/anonymized_cmc.csv", SCANS_PATH: "data/pre-processed/anonymized_cmc_png"}

# Which datasets to evaluate and save
RUN_CANADA = True                    # TRAIN_METRICS and VALIDATION (HN-PET-CT)
RUN_HN1    = True                    # HN1_TEST
RUN_CMC    = False                   # CMC_EXTERNAL

TWO_YEARS, FOUR_YEARS = 2 * 365, 4 * 365

# (run name, event, time frame, weights file, uses clinical data)
RUNS = [
    ############# Exact Reproducibility
    # ("dm_img",  DM,  TWO_YEARS,  "model_dm_689.pth.tar",     False),
    # ("lrf_img", LRF, TWO_YEARS,  "model_lrf_2340.pth.tar",   False),
    # ("os_img",  OS,  FOUR_YEARS, "model_os_1341.pth.tar",    False),
    # ("dm_cd",   DM,  TWO_YEARS,  "model_dm_cd_646.pth.tar",  True),
    # ("lrf_cd",  LRF, TWO_YEARS,  "model_lrf_cd_964.pth.tar", True),
    # ("os_cd",   OS,  FOUR_YEARS, "model_os_cd_2445.pth.tar", True),

    ############# Statistical Reproducibility
    # ("dm_img",  DM,  TWO_YEARS,  "model_dm_512.pth.tar",     False),
    # ("lrf_img", LRF, TWO_YEARS,  "model_lrf_2389.pth.tar",   False),
    # ("os_img",  OS,  FOUR_YEARS, "model_os_2090.pth.tar",    False),
    # ("dm_cd",   DM,  TWO_YEARS,  "model_dm_cd_582.pth.tar",  True),
    # ("lrf_cd",  LRF, TWO_YEARS,  "model_lrf_cd_964.pth.tar", True),
    # ("os_cd",   OS,  FOUR_YEARS, "model_os_cd_1442.pth.tar", True),
    # ("os_ann_cd",   OS,  FOUR_YEARS, "model_os_ANN_cd_308.pth.tar", True),

    ############# Statistical Reproducibility

    #################################### (5-fold CV, DM)
    # clinical (ANN)
    ("dm_ann_fold0", DM, TWO_YEARS, "model_dm_ANN_cd_5-fold_cv_fold0_163.pth.tar",  True, ANN),
    ("dm_ann_fold4", DM, TWO_YEARS, "model_dm_ANN_cd_5-fold_cv_fold4_137.pth.tar", True, ANN),
    
    # imaging (CNN)
    ("dm_img_fold0", DM, TWO_YEARS, "model_dm_5-fold_cv_fold0_370.pth.tar", False, CNN),
    ("dm_img_fold3", DM, TWO_YEARS, "model_dm_5-fold_cv_fold3_286.pth.tar", False, CNN),
    ("dm_img_fold4", DM, TWO_YEARS, "model_dm_5-fold_cv_fold4_419.pth.tar", False, CNN),
    
    # clinical + imaging (CNN)
    ("dm_cd_fold0", DM, TWO_YEARS, "model_dm_cd_5-fold_cv_fold0_282.pth.tar",  True, CNN),
    ("dm_cd_fold3", DM, TWO_YEARS, "model_dm_cd_5-fold_cv_fold3_436.pth.tar", True, CNN),
    ("dm_cd_fold4", DM, TWO_YEARS, "model_dm_cd_5-fold_cv_fold4_486.pth.tar",   True, CNN),

    #################################### (5-fold CV, LRF)
    # clinical (ANN)
    # ("lrf_ann_fold3", LRF, TWO_YEARS, "model_cmc_lrf_ANN_5-fold_cv_fold3_1008.pth.tar", True, ANN),
    
    # imaging (CNN)
    ("lrf_img_fold0", LRF, TWO_YEARS, "model_lrf_5-fold_cv_fold0_670.pth.tar",  False, CNN),
    ("lrf_img_fold2", LRF, TWO_YEARS, "model_lrf_5-fold_cv_fold2_481.pth.tar", False, CNN),
    ("lrf_img_fold3", LRF, TWO_YEARS, "model_lrf_5-fold_cv_fold3_498.pth.tar",  False, CNN),
    ("lrf_img_fold4", LRF, TWO_YEARS, "model_lrf_5-fold_cv_fold4_30.pth.tar",  False, CNN),
    
    # clinical + imaging (CNN)
    ("lrf_cd_fold0", LRF, TWO_YEARS, "model_lrf_cd_5-fold_cv_fold0_614.pth.tar", True, CNN),

    #################################### (5-fold CV, OS)
    # clinical (ANN)
    ("os_ann_fold0", OS, TWO_YEARS, "model_os_ANN_cd_5-fold_cv_fold0_109.pth.tar", True, ANN),
    ("os_ann_fold2", OS, TWO_YEARS, "model_os_ANN_cd_5-fold_cv_fold2_132.pth.tar", True, ANN),

    # imaging (CNN)
    ("os_img_fold0", OS, TWO_YEARS, "model_os_5-fold_cv_fold0_408.pth.tar",  False, CNN),
    ("os_img_fold1", OS, TWO_YEARS, "model_os_5-fold_cv_fold1_453.pth.tar", False, CNN),
    ("os_img_fold2", OS, TWO_YEARS, "model_os_5-fold_cv_fold2_396.pth.tar",  False, CNN),
    ("os_img_fold4", OS, TWO_YEARS, "model_os_5-fold_cv_fold4_258.pth.tar",  False, CNN),

    # clinical + imaging (CNN)
    ("os_cd_fold0", OS, TWO_YEARS, "model_os_cd_5-fold_cv_fold0_7.pth.tar", True, CNN),
    ("os_cd_fold2", OS, TWO_YEARS, "model_os_cd_5-fold_cv_fold2_435.pth.tar", True, CNN),
    ("os_cd_fold4", OS, TWO_YEARS, "model_os_cd_5-fold_cv_fold4_1338.pth.tar", True, CNN),

    ############# External Validation (Cohort-Split)
    # ("dm_img",  DM,  TWO_YEARS,  "model_dm_489.pth.tar",     False),
    # ("lrf_img", LRF, TWO_YEARS,  "model_cmc_lrf_1359.pth.tar",   False),
    # ("dm_cd",   DM,  TWO_YEARS,  "model_dm_cd_579.pth.tar",  True),
    # ("lrf_cd",  LRF, TWO_YEARS,  "model_cmc_lrf_cd_1086.pth.tar", True),

    ############# External Validation (5-fold CV, DM)
    # clinical (ANN)
    # ("dm_ann_fold0", DM, TWO_YEARS, "model_dm_ANN_cd_5-fold_cv_fold0_217.pth.tar",  True, ANN),
    # ("dm_ann_fold2", DM, TWO_YEARS, "model_dm_ANN_cd_5-fold_cv_fold2_2237.pth.tar", True, ANN),
    # ("dm_ann_fold3", DM, TWO_YEARS, "model_dm_ANN_cd_5-fold_cv_fold3_4.pth.tar",    True, ANN),
    # ("dm_ann_fold4", DM, TWO_YEARS, "model_dm_ANN_cd_5-fold_cv_fold4_130.pth.tar",  True, ANN),
    
    # imaging (CNN)
    # ("dm_img_fold0", DM, TWO_YEARS, "model_dm_5-fold_cv_fold0_207.pth.tar", False, CNN),
    # ("dm_img_fold1", DM, TWO_YEARS, "model_dm_5-fold_cv_fold1_264.pth.tar", False, CNN),
    # ("dm_img_fold2", DM, TWO_YEARS, "model_dm_5-fold_cv_fold2_190.pth.tar", False, CNN),
    # ("dm_img_fold4", DM, TWO_YEARS, "model_dm_5-fold_cv_fold4_511.pth.tar", False, CNN),
    
    # clinical + imaging (CNN)
    # ("dm_cd_fold2", DM, TWO_YEARS, "model_dm_cd_5-fold_cv_fold2_24.pth.tar",  True, CNN),
    # ("dm_cd_fold3", DM, TWO_YEARS, "model_dm_cd_5-fold_cv_fold3_554.pth.tar", True, CNN),
    # ("dm_cd_fold4", DM, TWO_YEARS, "model_dm_cd_5-fold_cv_fold4_8.pth.tar",   True, CNN),
 
    ############# 5-fold CV, LRF
    # ("lrf_ann_fold3", LRF, TWO_YEARS, "model_cmc_lrf_ANN_5-fold_cv_fold3_1008.pth.tar", True, ANN),
    # ("lrf_img_fold0", LRF, TWO_YEARS, "model_cmc_lrf_5-fold_cv_fold0_264.pth.tar",  False, CNN),
    # ("lrf_img_fold2", LRF, TWO_YEARS, "model_cmc_lrf_5-fold_cv_fold2_1382.pth.tar", False, CNN),
    # ("lrf_img_fold3", LRF, TWO_YEARS, "model_cmc_lrf_5-fold_cv_fold3_619.pth.tar",  False, CNN),
    # ("lrf_cd_fold0", LRF, TWO_YEARS, "model_cmc_lrf_cd_5-fold_cv_fold0_1149.pth.tar", True, CNN),
    # ("lrf_cd_fold3", LRF, TWO_YEARS, "model_cmc_lrf_cd_5-fold_cv_fold3_852.pth.tar",  True, CNN),
]

# Predictions and AUC do not depend on any seed (no augmentation, model.eval(), no shuffling).
# Only the bootstrap 95% CI uses random numbers (np.random in hn_cnn/metrics.py),
# so numpy is seeded to make the CI bounds repeatable.
random.seed(7651962)
np.random.seed(7651962)
torch.manual_seed(775135)


################################## Helpers ##################################


os.makedirs(RESULTS_DIR, exist_ok=True)
SUMMARY = os.path.join(RESULTS_DIR, "summary.txt")
 
 
def log(text=""):
    """Write to summary.txt only (nothing is printed to the terminal)."""
    with open(SUMMARY, "a") as f:
        f.write(text + "\n")
 
 
def find_weights(name):
    """Look for the weights file in WEIGHTS_DIR and WEIGHTS_DIR/with_clinical_data."""
    for folder in (WEIGHTS_DIR, os.path.join(WEIGHTS_DIR, "with_clinical_data")):
        path = os.path.join(folder, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"{name} not found in {WEIGHTS_DIR}")
 
 
def make_loader(paths, ids, event, timeframe):
    dataset = ImageDataset(
        paths[CLINICAL_DATA_PATH], paths[SCANS_PATH], TRANSFORM_TO_TENSOR,
        ids_to_use=ids, timeframe=timeframe, event=event,
    )
    loader = DataLoader(dataset, batch_size=len(dataset.keys), shuffle=False, drop_last=False)
    return loader
 
 
def load_model(weights, clinical, model_type=CNN):
    model = MODELS[model_type](clinical_data=clinical)
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

################################## Run info ##################################

with open(os.path.join(RESULTS_DIR, "run_info.txt"), "w") as f:
    f.write(f"python  {sys.version}\n")
    f.write(f"torch   {torch.__version__}\n")
    f.write(f"numpy   {np.__version__}\n")
    f.write(f"pandas  {pd.__version__}\n")
    f.write(f"platform {platform.platform()}\n")
    f.write(f"cuda available: {torch.cuda.is_available()}\n")

open(SUMMARY, "w").close()

################################## Main loop ##################################

RAW = open(os.path.join(RESULTS_DIR, "raw_output.txt"), "w")
try:
    with contextlib.redirect_stdout(RAW), contextlib.redirect_stderr(RAW):
        for entry in RUNS:
            run, event, timeframe, weights_name, clinical = entry[:5]
            model_type = entry[5] if len(entry) > 5 else CNN
            weights = find_weights(weights_name)
            log("=" * 90)
            log(f"RUN {run} | event {event} | time frame {timeframe} days | weights {weights_name} | clinical {clinical} | model {model_type}")
 
            sets = {}
 
            if RUN_CANADA:
                # Training and validation patients come from the HN-PET-CT (canada) file
                all_canada = ImageDataset(
                    CANADA[CLINICAL_DATA_PATH], CANADA[SCANS_PATH], TRANSFORM_TO_TENSOR,
                    timeframe=timeframe, event=event,
                )
                train_ids = [i for i in all_canada.keys if "HGJ" in i or "CHUS" in i]
                validation_ids = [i for i in all_canada.keys if "HMR" in i or "CHUM" in i]
                sets["TRAIN_METRICS"] = make_loader(CANADA, train_ids, event, timeframe)
                sets["VALIDATION"] = make_loader(CANADA, validation_ids, event, timeframe)
 
            if RUN_HN1:
                sets["HN1_TEST"] = make_loader(MAASTRO, [], event, timeframe)
 
            if RUN_CMC:
                sets["CMC_EXTERNAL"] = make_loader(CMC, [], event, timeframe)
 
            model = load_model(weights, clinical, model_type)
 
            # predict() only evaluates the sets in the dictionary; each set is stored under its own name
            output = predict(model, sets)
 
            for name, result in output.items():
                dataset = sets[name].dataset
                ids, labels = dataset.keys, dataset.y
                probs = np.ravel(result[PREDICTIONS])
                assert len(ids) == len(labels) == len(probs)
                ci = result[ROC][CI]
                log(f"  {name:14s} n={len(ids):4d} events={int(sum(labels)):3d} "
                    f"AUC={result[ROC][AUC]:.3f} "
                    f"(95% CI {ci[LOWER_BOUND]:.3f}-{ci[HIGHER_BOUND]:.3f}) "
                    f"loss={float(result[LOSS]):.3f}")
                pd.DataFrame({"id": ids, "label": labels, "prob": probs}).to_csv(
                    os.path.join(RESULTS_DIR, f"predictions_{run}_{name}.csv"), index=False)
 
except Exception as e:
    RAW.close()
    print(f"FAILED: {type(e).__name__}: {e}  (details in {RESULTS_DIR}/raw_output.txt)")
    raise SystemExit(1)
 
log("=" * 90)
log(f"Done. Files are in {RESULTS_DIR}")
RAW.close()
print("Done")