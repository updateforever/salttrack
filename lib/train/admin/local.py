from pathlib import Path


def _project_root() -> Path:
    # local.py -> admin -> train -> lib -> repo root
    return Path(__file__).resolve().parents[3]


def _dataset_root() -> Path:
    return Path("/data/DATASETS_PUBLIC/")


class EnvironmentSettings:
    def __init__(self):
        prj_dir = _project_root()
        data_root = _dataset_root()

        self.workspace_dir = str(prj_dir)
        self.tensorboard_dir = str(prj_dir / "tensorboard")
        self.pretrained_networks = str(prj_dir / "resource" / "pretrained_models")

        # rgb
        self.got10k_dir = str(data_root / "GOT10K" / "train") # /root/user-data/PUBLIC_DATASETS/GOT10K/train/
        self.got10k_val_dir = str(data_root / "GOT10K" / "val")
        self.got10k_lmdb_dir = str(data_root / "got10k_lmdb")

        self.lasot_dir = str(data_root / "lasot")
        self.lasot_lmdb_dir = str(data_root / "lasot_lmdb")

        self.trackingnet_dir = str(data_root / "trackingnet")
        self.trackingnet_lmdb_dir = str(data_root / "trackingnet_lmdb")

        self.coco_lmdb_dir = str(data_root / "coco_lmdb")
        self.coco_dir = str(data_root / "coco")
        self.ref_coco_dir = str(data_root / "refcoco")
        self.otb99_dir = str(data_root / "OTB_lang")

        self.tnl2k_dir = str(data_root / "TNL2K" / "TNL2K_train_subset")
        self.vasttrack_dir = str(data_root / "vasttrack" / "unisot_train_final_backup")
        
        # rgbt
        self.lasher_dir = str(data_root / "LasHeR" / "trainingset")

        # rgbe
        self.visevent_dir = str(data_root / "VisEvent_dataset" / "train_subset")

        # rgbd
        self.depthtrack_dir = str(data_root / "depthtrack_train")
