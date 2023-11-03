from .google_scanned_objects import *
from .realestate import *
from .deepvoxels import *
from .realestate import *
from .llff import *
from .llff_test import *
from .ibrnet_collected import *
from .realestate import *
from .spaces_dataset import *
from .nerf_synthetic import *
from .shiny import *
from .llff_render import *
from .shiny_render import *
from .nerf_synthetic_render import *
from .nmr_dataset import *
from .rffr import RFFRDataset
from .rffr_test import RFFRTestDataset
from .scannet_dataset import ScannetTrainDataset, ScannetValDataset
from .replica_dataset import ReplicaTrainDataset, ReplicaValDataset

dataset_dict = {
    "spaces": SpacesFreeDataset,
    "google_scanned": GoogleScannedDataset,
    "realestate": RealEstateDataset,
    "deepvoxels": DeepVoxelsDataset,
    "nerf_synthetic": NerfSyntheticDataset,
    "llff": LLFFDataset,
    "ibrnet_collected": IBRNetCollectedDataset,
    "llff_test": LLFFTestDataset,
    "shiny": ShinyDataset,
    "llff_render": LLFFRenderDataset,
    "shiny_render": ShinyRenderDataset,
    "nerf_synthetic_render": NerfSyntheticRenderDataset,
    "nmr": NMRDataset,
    # "rffr": RFFRDataset, 
    "rffr": RFFRTestDataset,
    "train_scannet": ScannetTrainDataset,  # for train semanitc segmentation
    "val_scannet": ScannetValDataset,  # for val semanitc segmentation
    "train_replica": ReplicaTrainDataset,  # for train semanitc segmentation
    "val_replica": ReplicaValDataset,  # for val semanitc segmentation
}
