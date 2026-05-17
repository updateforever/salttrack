from pathlib import Path

from lib.test.evaluation.environment import EnvSettings


def _project_root() -> Path:
    # local.py -> evaluation -> test -> lib -> repo root
    return Path(__file__).resolve().parents[3]


def _dataset_root() -> Path:
    return Path("/data/DATASETS_PUBLIC")


def local_env_settings():
    settings = EnvSettings()

    prj_dir = _project_root()
    data_root = _dataset_root()
    output_root = prj_dir / "output"

    settings.prj_dir = str(prj_dir)

    # Model / checkpoint paths
    settings.checkpoints_path = str(output_root / "checkpoints")
    settings.network_path = settings.checkpoints_path

    # Evaluation output paths
    settings.save_dir = str(output_root)
    settings.results_path = str(output_root / "test" / "tracking_results")
    settings.result_plot_path = str(output_root / "test" / "result_plots")
    settings.segmentation_path = str(output_root / "test" / "segmentation_results")

    # Dataset roots
    settings.davis_dir = str(data_root / "davis")
    settings.got10k_lmdb_path = str(data_root / "got10k_lmdb")
    settings.got10k_path = str(data_root / "got10k")
    settings.got_packed_results_path = ""
    settings.got_reports_path = ""
    settings.itb_path = str(data_root / "itb")
    settings.lasot_extension_subset_path = str(data_root / "lasot_extension_subset")
    # mhf
    settings.lasot_lmdb_path = str(data_root / "lasot_lmdb")
    settings.lasot_path = str(data_root / "lasot")
    settings.lasotlang_path = str(data_root / "lasot")
    settings.nfs_path = str(data_root / "nfs")
# mhf
    # 1. 指向存放 100 条视频序列图片的文件夹
    settings.otb_path = str(data_root / "OTB_lang" / "OTB_videos")

    # 2. 指向存放文本描述（JSON 或 TXT）的父目录
    # 注意：脚本通常会自动在下面寻找 OTB_query_test 文件夹
    settings.otblang_path = str(data_root / "OTB_lang")
    # settings.otb_lang_path = str(data_root / "OTB_lang" / "OTB_query_test")
    # settings.otb_path = str(data_root / "OTB_lang" / "OTB_query_test")
    settings.tc128_path = str(data_root / "TC128")
    settings.tn_packed_results_path = ""
    settings.tnl2k_path = str(data_root / "TNL2K" / "TNL2K_test_subset")
    settings.tpl_path = ""
    settings.trackingnet_path = str(data_root / "trackingnet")

    return settings  
