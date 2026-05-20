# parse_data.py — Detailed, line-by-line comments explaining how imaging and clinical data are loaded and prepared.

# --- Imports -------------------------------------------------------------

import os                             # Provides file path utilities (e.g., os.path.join)
import pandas as pd                   # Pandas for reading clinical data CSVs
from PIL import Image                 # Python Imaging Library for loading 2D image files (.png)
import torch                          # PyTorch for tensor operations and Dataset class

from hn_cnn.constants import *        # Project-wide constants: e.g., ID, TRAIN, EVENT keys, etc.
from hn_cnn.image_preprocessing import process_scan  # Function to preprocess 3D scan volumes (NIfTI)


# --- Clinical Data Mapping ----------------------------------------------

# Defines how each clinical variable in the CSV should be encoded for the model.
# Include each variable: Primary Site, T-stage, N-stage, ...
#    Key: the name of the column containing the variable in the dataset
#    Values: mapping between each output value and the codes used in the dataset
#    Thresholds: specify the thresholds when using a numeric value that needs to be discretize

CLINICAL_DATA = {
    # 1) Primary site of tumor
    PRIMARY_SITE: {
        KEY: "site",                 # CSV column name holding primary site
        VALUES: {                     # One-hot mapping from textual values to model codes
            "h": ["HYPOPHARYNX"],
            "l": ["LARYNX"],
            "n": ["NASOPHARYNX"],
            "o": ["OROPHARYNX"],
            "u": ["UNKNOWN"],
        }
    },

    # 2) T-stage (tumor size/extent)
    T_STAGE: {
        KEY: "tstage",              # CSV column name
        VALUES: {                     # Categorical mapping to t1/t2/... bins
           "t1": ['T1', '1'],
           "t2": ['T2', '2'],
           "t3": ['T3', '3', 'T3 (2)'],
           "t4": ['T4', 'T4A', 'T4B', '4'], 
           "tx": ['TX'],
        }
    },

    # 3) N-stage (lymph node involvement)
    N_STAGE: {
        KEY: "nstage",
        VALUES: {
            "n0": ['N0', '0'],
            "n1": ['N1', '1'],
            # In our analysis, N2 and N3 were groupped due to the low number
            # of patients with the N3 stage in the training set
            "n2": ['N2', 'N2A', 'N2B', 'N2C', '2', 'N3', 'N3B', '3'],
            "n3": ['N3', 'N3B', '3'],
        }
    },

    # 4) TNM overall stage grouping
    TNM_STAGE: {
        KEY: "groupstage",
        VALUES: {  # Map all textual stage labels to s1/s2/... bins
            "s1": ['STADE I', 'STAGE I', 'I'],
            "s2": ['STADE II', 'STAGE II', 'STAGEII', 'STAGE IIB', 'II', 'IIB'],
            "s2b": ['STAGE IIB', 'IIB'],
            "s3": ['STADE III', 'STAGE III', 'III'],
            "s4": ['STAGE IV', 'STADE IV', 'STAGE IVA', 'STAGE IVB', 'STADE IVA', 'STADE IVB','IV', 'IVA', 'IVB', 'IVC'],
            "s4o": ['STAGE IV', 'STADE IV', 'IV'],
            "s4a": ['STAGE IVA', 'STADE IVA', 'IVA'],
            "s4b": ['STAGE IVB', 'STADE IVB', 'IVB'],
            "s4c": ['IVC'],
            "nan": ["N/A", "NAN"],
        }
    },

    # 5) HPV status (binary positive/negative)
    HPC: {
        KEY: "overall_hpv_status",
        VALUES: {
            "hpc+": ['+', 'POSITIVE'],
            "hpc-": ['-', 'NEGATIVE'],
            "nan": ['N/A', 'NAN', 'NOT TESTED'],
        }
    },

    # 6) Tumor volume (continuous numeric → discretized into bins)
         # To calculate the volume we used FSL:
         # /usr/share/fsl/5.0/bin/fslstats /path/to/masks/{mask_id}_mask.nii.gz -V
    VOLUME: {
        KEY: "vol",                  # CSV column with numeric volume
        THRESHOLDS: {                 # Bins defined by min/max values (in mm^3)
            "vol0": [0, 11000],
            "vol1": [11000, 24000],
            "vol2": [24000, 43000],
            "vol3": [43000, -1]      # -1 means no upper bound
        }
    },

    # 7) Largest cross-sectional area (continuous → bins)
         # To calculate the area:
         # cord = list(scan.header.get_zooms())[0:1]
         # area.append(pixels_by_slice[np.argmax(pixels_by_slice)] * cord[0] * cord[1])
    AREA: {
        KEY: "area",
        THRESHOLDS: {
            "area0": [0, 470],
            "area1": [470, 770],
            "area2": [770, 1100],
            "area3": [1100, -1]
        }
    }
}


# --- Event Definitions --------------------------------------------------

# Maps clinical CSV columns for each outcome to the internal keys used in code.
EVENTS = {
    DM: {  # Distant metastasis
        EVENT: ["Distant", "event_distant_metastases"],
        TIME_TO_EVENT: ["Time - diagnosis to DM (days)", "distant_metastases_in_days"],
        FU: ["Time - diagnosis to last follow-up(days)", "distant_metastases_in_days"],
    },
    LRF: {  # Locoregional recurrence
        EVENT: ["Locoregional", "event_locoregional_recurrence"],
        TIME_TO_EVENT: ["Time - diagnosis to LR (days)", "locoregional_recurrence_in_days"],
        FU: ["Time - diagnosis to last follow-up(days)", "locoregional_recurrence_in_days"],
    },
    OS: {  # Overall survival
        EVENT: ["Death", "event_overall_survival"],
        TIME_TO_EVENT: ["Time - diagnosis to Death (days)", "overall_survival_in_days"],
        FU: ["Time - diagnosis to last follow-up(days)", "overall_survival_in_days"],
    },
}

# Primary site
# ['HYPOPHARYNX', 'LARYNX', 'NASOPHARYNX', 'OROPHARYNX', 'UNKNOWN']
# T-stage
# ['T1', 'T2', 'T3', 'T4', 'T4A', 'T4B', 'TX']
# N-stage
# ['N0', 'N1', 'N2', 'N2B', 'N2C', 'N3', 'N3B']
# TNM-stage
# ['STAGE I', 'STAGE II', 'STAGE IIB', 'STAGE III', 'STAGE IV', 'STAGE IVA', 'STAGE IVB']
# HPV Status
# ['HPV+', 'HPV-', 'UNKNOWN']


# --- parse_clinical() ----------------------------------------------------

def parse_clinical(tabular):
    """ Convert one row of clinical data into a FloatTensor of selected features.
    """
    # Iterate each clinical variable mapping
    for variable, variable_info in CLINICAL_DATA.items():
        column_name = variable_info[KEY]            # CSV column name for this variable

        # 1) Handle categorical variables via VALUES mapping
        if VALUES in variable_info:
            column_value = str(tabular[column_name]).upper().strip()  # Normalize value
            variable_values = []  # Track all accepted strings
            
            # One-hot encode: create a new binary column per category key
            for site_key, site_values in variable_info[VALUES].items():
                variable_values.extend(site_values)
                tabular[site_key] = int(column_value in site_values)
            
            # Warn if unknown category encountered
            if column_value not in variable_values:
                print(f'Error in parsing the {variable}, unknow value: {tabular[column_name]}')
        
        # 2) Handle numeric binning via THRESHOLDS mapping
        elif THRESHOLDS in variable_info:
            # Variables that need to be discretized
            column_value = float(tabular[column_name])
            for site_key, site_values in variable_info[THRESHOLDS].items():
                # Assign to bin if within [low, high)
                tabular[site_key] = int(
                    column_value >= site_values[0] and (site_values[1] < 0 or column_value < site_values[1])
                )
    # Features included in the training
    features = tabular[["n0", "n1", "n2", "t1", "t2", "t3", "t4", "vol0", "vol1", "vol2", "vol3"]]
    return torch.FloatTensor(features.tolist())  # Convert to Tensor


# --- parse_event() ------------------------------------------------------

def parse_event(tabular, event):
    """ Extract event label and time-to-event from one row.
    """
    parsed_event = {}
    
    # For each EVENT, TIME_TO_EVENT, FU key, pick the first matching CSV column
    for variable, variable_info in EVENTS[event].items():
        for column in variable_info:
            if column in tabular:
                parsed_event[variable] = tabular[column]
                break
    # TODO: Create error when one of the parameters doesn't have a result
    return parsed_event


# --- ImageDataset Class -----------------------------------------------

class ImageDataset(torch.utils.data.Dataset):
    """Class to read, transform, and provide the data to the CNN
    """

    def __init__(
        self,
        dataset_path,           # Path to clinical CSV
        scans_path,             # Folder of scan images (.png) or volumes
        transforms,             # torchvision transforms or tensor conversion
        ids_to_use = [],        # List of specific patient IDs to include (empty = all)
        timeframe = 2*365,      # Max follow-up days to include
        event = "lrf",           # Which event type (dm, lrf, os)
        preprocess = False,     # Whether to preprocess NIfTI volumes instead of PNG
        mask_path = None,
        mask_suffix = None
    ):

        # Read clinical CSV; delimiter is semicolon
        # self.tabular = pd.read_csv(dataset_path, delimiter=';')
        self.tabular = pd.read_csv(dataset_path)

        # Extract the information
        self.transforms = transforms
        self.images = {}   # {id: PIL.Image or 3D array}
        self.tab_data = {} # {id: clinical tensor}
        self.keys = []     # List of patient IDs in dataset
        self.y = []        # Binary labels per ID

        # Loop through each patient record
        for index, tb in self.tabular.iterrows():
            event_info = parse_event(tb, event)    # Get event result and follow-up time
            id = tb[ID]                            # Unique patient identifier
            # Include ID if in ids_to_use (or include all) and within timeframe or event occurred
            if (len(ids_to_use) == 0 or id in ids_to_use) and \
                (event_info[FU] >= timeframe or event_info[EVENT] == 1):
                    
                # Load image: either preprocess 3D scan or open 2D PNG
                    if preprocess and mask_path:
                        self.images[id] = process_scan(
                            scan_path=f"{scans_path}/{id}.nii.gz",
                            mask_path=f"{mask_path}/{id}{mask_suffix}.nii.gz",
                        )
                    
                    else:
                        self.images[id] = Image.open(f"{scans_path}/{id}.png")
                    
                    # Parse clinical features into a tensor
                    self.tab_data[id] = parse_clinical(tb)
                    self.keys.append(id) # Track ID order

                    # Label: 1 if event occurred within timeframe, else 0
                    self.y.append(
                        1 if event_info[EVENT] == 1 and event_info[TIME_TO_EVENT] <= timeframe else 0
                    )

        # Summary of loaded data
        print(f"Imported {len(self.images)} scans from {scans_path} with {sum(self.y)} events")

    def __len__(self):
        # Number of samples in dataset
        return len(self.keys)

    def __getitem__(self, idx):
        # Retrieve one sample by index (int or tensor index)
         # Convert tensor index to int if needed
         # Get the patient ID from the keys list
         # Apply transformations to the image
         # Return the image, clinical data, and label as a tuple
        if torch.is_tensor(idx):
            idx = idx.tolist()
        id = self.keys[idx]
        image = self.transforms(self.images[id])
        return image, self.tab_data[id], int(self.y[idx])
