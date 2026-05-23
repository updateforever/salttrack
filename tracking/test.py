import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from lib.test.evaluation import get_dataset
from lib.test.evaluation.running import run_dataset
from lib.test.evaluation.tracker import Tracker, trackerlist
from lib.test.evaluation.environment import env_settings

def run_tracker(tracker_name, tracker_param, run_id=None, dataset_name='otb', sequence=None, debug=0, threads=0,
                ckpt_path="", num_gpus=8, run_tag=None):
    """Run tracker on sequence or dataset.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        run_id: The run id.
        dataset_name: Name of dataset (otb, nfs, uav, tpl, vot, tn, gott, gotv, lasot).
        sequence: Sequence number or name.
        debug: Debug level.
        threads: Number of threads.
    """

    dataset = get_dataset(dataset_name)

    # dataset = dataset[:2]
    if sequence is not None:
        dataset = [dataset[sequence]]


    env = env_settings()
    checkpoints_path = ckpt_path

    results_dir = os.path.join(env.save_dir, tracker_param)
    os.makedirs(results_dir, exist_ok=True)
    # trackers = [Tracker(tracker_name, tracker_param, dataset_name, run_id)]

    trackers = trackerlist(name=tracker_name, parameter_name=tracker_param, dataset_name=dataset_name,
                           run_ids=run_id)

    result_bucket = run_tag if run_tag else dataset_name
    results_dir_item = os.path.join(results_dir, result_bucket)

    trackers[0].results_eval_dir = results_dir_item
    trackers[0].results_dir = results_dir_item
    trackers[0].checkpoints_path = checkpoints_path

    print("checkpoints_path_item:", checkpoints_path)
    print("results_dir_item:", results_dir_item)
    run_dataset(dataset, trackers, debug, threads, num_gpus=num_gpus)


def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    parser = argparse.ArgumentParser(description='Run tracker on sequence or dataset.')
    parser.add_argument('--tracker_name', type=str, default="salttrack", help='Name of tracking method.')
    parser.add_argument('--tracker_param', type=str, default="salttrack_base", help='Name of config file.')    
    parser.add_argument('--runid', type=int, default=None, help='The run id.')
    parser.add_argument('--dataset_name', type=str, default="tnl2k", help='Name of dataset (lasot_lang , tnl2k, otb99_lang, lasot_extension_subset_lang, videocube_test_tiny).')
    parser.add_argument('--ckpt_path', type=str, default="./SALTTrack_b.pth.tar",
                        help='Name of dataset (lasot_lang , tnl2k, otb99_lang, lasot_extension_subset_lang, videocube_test_tiny).')

    parser.add_argument('--sequence', type=str, default=None, help='Sequence number or name.')
    parser.add_argument('--debug', type=int, default=0, help='Debug level.')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads.')
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--run_tag', type=str, default=None,
                        help='Optional subdirectory name for saving evaluation results.')


    args = parser.parse_args()

    try:
        seq_name = int(args.sequence)
    except:
        seq_name = args.sequence

    run_tracker(args.tracker_name, args.tracker_param, args.runid, args.dataset_name, seq_name, args.debug,
                args.threads, args.ckpt_path, num_gpus=args.num_gpus, run_tag=args.run_tag)


if __name__ == '__main__':
    main()
