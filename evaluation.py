import os
from absl import app, flags
import numpy as np
from tqdm import tqdm
from colorama import Fore, Style
from pathlib import Path
import torch

from datasets import TFRecordMotionDataset
from visualization import H36MForwardKinematics, CMUForwardKinematics, Visualizer
from metrics import MetricsEngine
from utils import load_state, select_device
import flag_sets
FLAGS = flags.FLAGS



##############################################
def evaluate_model(test_data, model, metrics_engine, device, one_batch_run=False, db=None):
    model.eval()
    eval_result = dict()
    metrics_engine.reset()
    
    D = 2 if FLAGS.even else 1
    windows_size = FLAGS.input_seq_len + FLAGS.target_seq_len
    input_seq_len = FLAGS.input_seq_len//D
    inds = range(0, windows_size, D)
    nb = len(test_data)
    test_samples = test_data.get_tf_samples()
    pbar = enumerate(test_samples)
    pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}')

    with torch.no_grad():
        for _, batch in pbar:
            data_id = batch['id']
            inputs = batch['inputs'][:, inds, :]
            xy = torch.tensor(inputs).to(device)
            x = xy[:, :input_seq_len].permute(0, 2, 1)
            y = xy[:, input_seq_len:]
            seed_sequence = xy[:, :input_seq_len]

            _, pred_motion = model(x)

            gt_motion = y.cpu().numpy()  
            pred_motion = pred_motion.cpu().numpy() 
            seed_sequence = seed_sequence.cpu().numpy()

            pred_motion = test_data.unnormalize_zero_mean_unit_variance_channel({'poses': pred_motion}, 'poses')
            gt_motion = test_data.unnormalize_zero_mean_unit_variance_channel({'poses': gt_motion}, 'poses')
            seed = test_data.unnormalize_zero_mean_unit_variance_channel({'poses': seed_sequence}, 'poses')
            metrics_engine.compute_and_aggregate(pred_motion['poses'], gt_motion['poses'])
            # Store each test sample and corresponding predictions with the sample IDs.
            if db == 'h36m':
                for i in range(pred_motion['poses'].shape[0]):
                    seq_name = data_id[i].decode('utf-8').split('_')[1]
                    if seq_name not in eval_result.keys():
                        eval_result[seq_name] = []
                    eval_result[seq_name].append(
                        (seed['poses'][i], pred_motion['poses'][i], gt_motion['poses'][i]))
            elif db == 'cmu':
                for i in range(pred_motion['poses'].shape[0]):
                    seq_name = data_id[i].decode('utf-8').split('/')[-1]
                    if seq_name not in eval_result.keys():
                        eval_result[seq_name] = []
                    eval_result[seq_name].append((seed['poses'][i], pred_motion['poses'][i], gt_motion['poses'][i]))
            else:
                raise Exception('Only cmu and h36m datasets are supported!')

            pbar.set_description('Processing sequence: {}'.format(seq_name))

            if one_batch_run:
                break

    # finalize the metrics computation
    final_metrics = {}
    seq_names = eval_result.keys()
    for _, k in enumerate(seq_names):
        metrics_engine.reset()
        for idx in range(len(eval_result[k])):
            pred = eval_result[k][idx][1]
            gt = eval_result[k][idx][2]
            pred = np.expand_dims(pred, axis=0)
            gt = np.expand_dims(gt, axis=0)

            metrics_engine.compute_and_aggregate(pred, gt)
        final_metrics[k] = metrics_engine.get_final_metrics()

    return final_metrics, eval_result


##############################################
def main(_argv):

    db = FLAGS.db
    exp_name = db + '_' + FLAGS.exp
    if db == 'cmu':
        FLAGS.num_joints = 24
    elif db == 'h36m':
        FLAGS.num_joints = 21
    else:
        raise Exception('Only cmu and h36m datasets are supported!')

    if 'main' in FLAGS.exp:
        from models import ResChunk
        
    elif 'ablation' in FLAGS.exp:
        abl_no = FLAGS.exp.split('_')[1]
        if abl_no == '1':
            from models_abl_1 import ResChunk
        elif abl_no == '2':
            from models_abl_2 import ResChunk
        elif abl_no == '3':
            from models_abl_3 import ResChunk
        elif abl_no == '4':
            abl_4_no = FLAGS.exp.split('_')[2]
            if abl_4_no == '1':
                from models_abl_4_1 import ResChunk
            if abl_4_no == '2':
                from models_abl_4_2 import ResChunk
            if abl_4_no == '3':
                from models_abl_4_3 import ResChunk
        elif abl_no == '5':
            from models_abl_5 import ResChunk
    else:
        raise Exception('Invalid experiment name!')

    print(f'{Fore.YELLOW}') 
    device = select_device(FLAGS.device, batch_size=FLAGS.batch_size)
    print(f'{Style.RESET_ALL}')

    data_dir = os.path.join(FLAGS.data_dir, db)
    eval_dir = Path(FLAGS.run_dir) / exp_name / 'test'
    if not os.path.isdir(eval_dir):
        os.mkdir(eval_dir)
    
    model = ResChunk(FLAGS).to(device) 
    if FLAGS.last_only:
        latest_checkpoint = Path(FLAGS.run_dir) / Path(exp_name) / 'checkpoints'/ 'last.pt.tar'
    else:
        latest_checkpoint = Path(FLAGS.run_dir) / Path(exp_name) / 'checkpoints'/  FLAGS.load_checkpoint+'.pt.tar'

    state = load_state(latest_checkpoint, cuda=False)
    model.load_state_dict(state['model'])
    del state

    window_length = FLAGS.input_seq_len + FLAGS.target_seq_len
    rep = FLAGS.data_type
    test_data_split = 'test_dynamic' if FLAGS.dynamic_test_split else 'test'
    test_data_path = os.path.join(data_dir, rep, test_data_split, db + '-?????-of-?????')
    meta_data_path = os.path.join(data_dir, rep, 'training', 'stats.npz')

    # Create dataset.
    test_data = TFRecordMotionDataset(data_path=test_data_path, meta_data_path=meta_data_path,
                                      batch_size=FLAGS.batch_size, shuffle=False,
                                      windows_size=window_length, window_type='from_beginning',
                                      num_parallel_calls=FLAGS.workers, normalize=FLAGS.normalize)

    print(f'{Fore.YELLOW}Evaluating Model: {Style.RESET_ALL}' + str(latest_checkpoint))

    # Create metrics engine
    target_seq_len_metric = FLAGS.target_seq_len // 2 if FLAGS.even else FLAGS.target_seq_len

    fk_engine = H36MForwardKinematics() if db == 'h36m' else CMUForwardKinematics()
    metric_target_lengths = FLAGS.METRIC_TARGET_LENGTHS
    
    target_lengths = [x for x in metric_target_lengths if x <= target_seq_len_metric]

    metrics_engine = MetricsEngine(fk_engine, target_lengths, which=['mpjpe'], rep=rep)
    # reset computation of metrics
    metrics_engine.reset()

    print(f'{Fore.YELLOW}Evaluating', db, 'test set...', f'{Style.RESET_ALL}')

    final_metrics, eval_result = evaluate_model(test_data, model, metrics_engine, device, 
                                                one_batch_run=FLAGS.one_batch, db=db)

    seq_names = final_metrics.keys()
    for _, k in enumerate(seq_names):
        test_metric = final_metrics[k]
        s = metrics_engine.get_summary_string_all(test_metric, target_lengths, 
                                            at_mode=True, report_pck=False, tb_writer=None, 
                                            step=0, training=False, train_loss=None, val_loss=None)
        
        print(f'{Fore.GREEN}', '********', k, '********', s, f'{Style.RESET_ALL}')
        print()

    if FLAGS.visualize:
        visualizer = Visualizer(db=db, interactive=FLAGS.interactive, fk_engine=fk_engine, rep=rep,
                                output_dir=eval_dir, skeleton=FLAGS.skel, dense=FLAGS.dense,
                                to_video=FLAGS.to_video, save_figs=FLAGS.save_figs)

        print(f'{Fore.YELLOW}Visualizing some samples...{Style.RESET_ALL}')

        seq_names = eval_result.keys()
        for _, k in enumerate(seq_names):
            seq = {}
            best = np.inf
            for i in range(len(eval_result[k])): 
                seq[k] = [eval_result[k][i]]
                r = best_sel(db, seq, target_lengths)  
                if r[24] < best:
                    best = r[24]
                    idx = i
            visualizer.visualize_results(
                    eval_result[k][idx][0], eval_result[k][idx][1], eval_result[k][idx][2], title=k+'_'+str(idx))




def best_sel(db, eval_result, target_lengths):
    target_lengths = [x for x in target_lengths if x <= 25]
    target_lengths = [x-1 for x in target_lengths]

    for act in eval_result.keys():
        n = 0
        per_frame_all = 0
        for seqs in eval_result[act]:
            if len(seqs[1].shape) > 2:
                predictions = torch.tensor(seqs[1])
                targets = torch.tensor(seqs[2])
            else:
                predictions = torch.tensor(seqs[1]).unsqueeze(0)
                targets = torch.tensor(seqs[2]).unsqueeze(0)

            batch_size = predictions.shape[0]
            seq_length = predictions.shape[1]

            predictions = predictions.view(batch_size, seq_length, -1, 3)
            targets = targets.view(batch_size, seq_length, -1, 3)

            mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(
                targets - predictions, dim=3), dim=2), dim=0)
            per_frame_all += mpjpe_p3d_h36.numpy()
            n += batch_size

        per_frame_all = (per_frame_all / n) * 1000
        return per_frame_all



if __name__ == '__main__':
    app.run(main)
