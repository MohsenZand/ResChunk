import os
from absl import app, flags
from pathlib import Path
import torch 
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from colorama import Fore, Style

from utils import select_device, increment_path, count_parameters, load_state
from datasets import TFRecordMotionDataset
from visualization import H36MForwardKinematics, CMUForwardKinematics
from metrics import MetricsEngine
import flag_sets 
FLAGS = flags.FLAGS



##############################################
class Trainer():
    def __init__(self):
        super().__init__()
        
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
        input_seq_len = FLAGS.input_seq_len
        target_seq_len = FLAGS.target_seq_len
        rep = FLAGS.data_type
        data_dir = os.path.join(FLAGS.data_dir, db)

        # dataset
        train_data_split = 'training_dynamic' if FLAGS.dynamic_train_split else 'training'
        val_data_split = 'validation_dynamic' if FLAGS.dynamic_val_split else 'validation'

        self.train_data_path = os.path.join(data_dir, rep, train_data_split, db + '-?????-of-?????')
        self.meta_data_path = os.path.join(data_dir, rep, 'training', 'stats.npz')
        self.val_data_path = os.path.join(data_dir, rep, val_data_split, db + '-?????-of-?????')

        model = ResChunk(FLAGS).to(device) 
        
        print(f'{Fore.YELLOW}') 
        print('number of param: {}'.format(count_parameters(model)))
        print(f'{Style.RESET_ALL}')
        
        params = model.parameters()
        optim = torch.optim.Adam(params, lr=FLAGS.lr, betas=FLAGS.betas, weight_decay=FLAGS.regularizer)       
        self.scheduler = None

        if FLAGS.load_weights:
            if FLAGS.last_only:
                latest_checkpoint = Path(FLAGS.run_dir) / Path(exp_name) / 'checkpoints'/ 'last.pt.tar'
            else:
                latest_checkpoint = Path(FLAGS.run_dir) / Path(exp_name) / 'checkpoints'/  FLAGS.load_checkpoint+'.pt.tar'
            cuda = False if device == 'cpu' else True 
            state = load_state(latest_checkpoint, cuda)
            optim.load_state_dict(state['optim'])
            model.load_state_dict(state['model'])
            FLAGS.steps = state['iteration'] + 1
            FLAGS.init_epoch = state['epoch'] + 1
            if self.scheduler is not None and state.get('scheduler', None) is not None:
                self.scheduler.load_state_dict(state['scheduler'])
            del state
            
            log_dir = Path(FLAGS.run_dir) / exp_name

        else:
            log_dir = increment_path(Path(FLAGS.run_dir) / exp_name, exist_ok=FLAGS.exist_ok)  
            
        checkpoints_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(checkpoints_dir):
            os.makedirs(checkpoints_dir)
        
        # Create metrics engine
        target_seq_len_metric = FLAGS.target_seq_len // 2 if FLAGS.even else FLAGS.target_seq_len

        fk_engine = H36MForwardKinematics() if db == 'h36m' else CMUForwardKinematics()
        metric_target_lengths = FLAGS.METRIC_TARGET_LENGTHS
        
        target_lengths = [x for x in metric_target_lengths if x <= target_seq_len_metric]
        
        metrics_engine = MetricsEngine(fk_engine, target_lengths, which=['mpjpe'], rep=rep)
        metrics_engine.reset()
        
        self.early_stopping = EarlyStopping(tolerance=5)
        self.model = model
        self.log_dir = log_dir
        self.windows_size = input_seq_len + target_seq_len
        self.checkpoints_dir = checkpoints_dir
        self.metrics_engine = metrics_engine
        self.optim = optim
        self.device = device
        self.tb_writer = SummaryWriter(log_dir=log_dir) 
        self.target_lengths = target_lengths
        self.fk_engine = fk_engine
        self.db = db
        self.global_step = FLAGS.steps

        
    def train(self):    
        device = self.device
        train_data_path = self.train_data_path
        val_data_path = self.val_data_path
        meta_data_path = self.meta_data_path
        windows_size = self.windows_size
        checkpoints_dir = self.checkpoints_dir
        batch_size = FLAGS.batch_size

        D = 2 if FLAGS.even else 1
        input_seq_len = FLAGS.input_seq_len//D
        inds = range(0, windows_size , D)

        train_data = TFRecordMotionDataset(data_path=train_data_path, meta_data_path=meta_data_path, 
                                            batch_size=batch_size, shuffle=True, 
                                            windows_size=windows_size, window_type='random', 
                                            num_parallel_calls=FLAGS.workers, normalize=FLAGS.normalize)
        
        val_data = TFRecordMotionDataset(data_path=val_data_path, meta_data_path=meta_data_path, 
                                            batch_size=batch_size, shuffle=True, 
                                            windows_size=windows_size, window_type='random', 
                                            num_parallel_calls=FLAGS.workers, normalize=FLAGS.normalize)
        nb = len(train_data)

        for epoch in range(FLAGS.init_epoch, FLAGS.num_epochs):
            self.model.train()
            
            print(f'{Fore.YELLOW}', 'Training on {}. Epoch/batch_size/device/lr: {}/{}/{}/{}'.format(
                    self.db, epoch, batch_size, device, FLAGS.lr), f'{Style.RESET_ALL}')
            
            train_samples = train_data.get_tf_samples()
            pbar = enumerate(train_samples)
            pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}') 
            epoch_loss = 0.0
            
            for _, batch in pbar:
                self.optim.zero_grad()

                inputs = batch['inputs'][:, inds, :]
                xy = torch.tensor(inputs).to(device)
                x = xy[:, :input_seq_len].permute(0,2,1)
                y = xy[:, input_seq_len:].permute(0,2,1)

                if x.shape[2] < y.shape[2]:
                    x = torch.cat((x, x[..., -1:].repeat(1,1,y.shape[2]-x.shape[2])), dim=2)

                loss, _ = self.model(x, y)

                # backward
                self.model.zero_grad()
                self.optim.zero_grad()
                loss.backward()
                    
                # operate grad
                if FLAGS.max_grad_clip > 0:
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), FLAGS.max_grad_clip)
                if FLAGS.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), FLAGS.max_grad_norm)

                self.optim.step()
                epoch_loss += loss.item()
                pbar.set_description('Iter/Epoch:{}/{}  Loss:{:.5f}'.format(self.global_step, epoch, loss.data))
                self.global_step += 1
                
            if self.scheduler is not None:
                self.scheduler.step()
                
            epoch_loss = float(epoch_loss / float(nb))

            print(f'{Fore.YELLOW}Validation is starting ... {Style.RESET_ALL}')
            
            self.model.eval()
            val_loss = self.validation(val_data)

            self.metrics_engine.train_loss = epoch_loss
            self.metrics_engine.val_loss = val_loss
            final_metrics = self.metrics_engine.get_final_metrics()
            s = self.metrics_engine.get_summary_string_all(final_metrics, self.target_lengths, at_mode=True, 
                                                        report_pck=False, tb_writer=self.tb_writer, step=epoch, 
                                                        training=True, train_loss=epoch_loss, val_loss=val_loss)
            print(f'{Fore.GREEN}', s, f'{Style.RESET_ALL}')
            
            self.metrics_engine.reset()

            print(f'{Fore.YELLOW}Validation finished!{Style.RESET_ALL}')

            if FLAGS.last_only:
                save_model(self.model, self.optim, self.scheduler, checkpoints_dir, epoch, self.global_step, last=True)
            else:
                save_model(self.model, self.optim, self.scheduler, checkpoints_dir, epoch, self.global_step, last=False)
            
            self.early_stopping(val_loss, self.model, self.optim, self.scheduler, checkpoints_dir, epoch, self.global_step)
            if self.early_stopping.early_stop:
                print("Early stopping met at epoch:", epoch)
                break

    def validation(self, val_data):
        device = self.device
        D = 2 if FLAGS.even else 1
        input_seq_len = FLAGS.input_seq_len//D
        inds = range(0, self.windows_size, D)  
        nb = len(val_data)
        val_samples = val_data.get_tf_samples()
        pbar = enumerate(val_samples)
        pbar = tqdm(pbar, total=nb, bar_format='{l_bar}{bar:50}{r_bar}{bar:-10b}')
        val_loss = 0.0
        
        with torch.no_grad():
            for bi, batch in pbar:
                inputs = batch['inputs'][:, inds, :]
                xy = torch.tensor(inputs).to(device)
                x = xy[:, :input_seq_len].permute(0,2,1)
                y = xy[:, input_seq_len:]

                if x.shape[2] < y.shape[1]:
                    x = torch.cat((x, x[..., -1:].repeat(1,1,y.shape[1]-x.shape[2])), dim=2)

                loss, pred_motion = self.model(x)

                if loss > 0:
                    val_loss += loss.data

                gt_motion = y.cpu().numpy()  
                pred_motion = pred_motion.cpu().numpy() 
                
                pred_motion = val_data.unnormalize_zero_mean_unit_variance_channel({'poses': pred_motion}, 'poses')
                gt_motion = val_data.unnormalize_zero_mean_unit_variance_channel({'poses': gt_motion}, 'poses')
                self.metrics_engine.compute_and_aggregate(pred_motion['poses'], gt_motion['poses'])
                
                pbar.set_description('Validation/Batch: {}  Loss:{:.5f}'.format(bi, loss.data))

        return val_loss / float(nb)




def save_model(model, optim, scheduler, dir, epoch, iteration, last=False):
    if last:
        path = os.path.join(dir, 'last.pt.tar')
    else:
        path = os.path.join(dir, 'checkpoint_{}.pth.tar'.format(iteration))
    state = {}
    state['iteration'] = iteration
    state['epoch'] = epoch
    state['model_name'] = model.__class__.__name__
    state['model'] = model.state_dict()
    state['optim'] = optim.state_dict()
    if scheduler is not None:
        state['scheduler'] = scheduler.state_dict()
    else:
        state['scheduler'] = None

    torch.save(state, path)


class EarlyStopping():
    def __init__(self, tolerance=10):

        self.tolerance = tolerance
        self.counter = 0
        self.early_stop = False
        self.last_loss = 0

    def __call__(self, validation_loss, model, optim, scheduler, checkpoints_dir, epoch, global_step):
        if (validation_loss - self.last_loss) > 0:
            self.counter +=1
            if self.counter >= self.tolerance:  
                self.early_stop = True
                save_model(self.model, self.optim, self.scheduler, checkpoints_dir, self.epoch, self.global_step, last=False)
        else:
            self.counter = 0
            self.model = model
            self.optim = optim 
            self.scheduler = scheduler 
            self.epoch = epoch
            self.global_step = global_step

        self.last_loss = validation_loss


def main(_):
    trainer = Trainer()
    trainer.train()



if __name__ == '__main__':
    app.run(main)
