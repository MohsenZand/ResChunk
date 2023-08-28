from absl import flags


# GNERAL 
flags.DEFINE_string('db', 'cmu', 'dataset name, cmu or h36m')
flags.DEFINE_string('exp', 'main', 'experiment name, main or ablation? if ablation, specify the number as ablation_no')
flags.DEFINE_string('data_dir', 'PATH', 'Where to store the tfrecords. Processed data path')
flags.DEFINE_string('run_dir', 'PATH', 'save to run_dir/db_exp')
flags.DEFINE_bool('exist_ok', True, 'existing name ok, do not increment')
flags.DEFINE_bool('load_weights', False, 'use pretrain weights, stored in the checkpoint path in exp dir')

# MODEL
flags.DEFINE_enum('data_type', 'aa', ['aa', 'rotmat', 'quat'],'Which data representation: rotmat (rotation matrix), aa (angle axis), quat (quaternion)')
flags.DEFINE_integer('input_seq_len', 50, 'n number of frames as a seed sequence. if even==True, n/2 is used')
flags.DEFINE_integer('target_seq_len', 50, 'n number of frames that model must predict, if even==True, n/2 frames are predicted')
flags.DEFINE_bool('even', True, 'select even frames, FPS/2')
flags.DEFINE_integer('num_joints', 0, 'it is set in train.py')

# TRAIN
flags.DEFINE_string('device', '0', 'cuda device, i.e. 0 or 0,1,2,3 or cpu')
flags.DEFINE_integer('batch_size', 128, 'batch_size')
flags.DEFINE_integer('num_epochs', 1000, 'num_epochs')
flags.DEFINE_integer('init_epoch', 0, 'init epoch is set when loading checkpoint')
flags.DEFINE_integer('steps', 0, 'steps is set when loading checkpoint')
flags.DEFINE_float('lr', 0.0002, 'learning rate')
flags.DEFINE_integer('workers', 1, 'maximum number of dataloader workers')
flags.DEFINE_multi_float('betas', [0.9, 0.9999], 'betas')
flags.DEFINE_float('regularizer', 0.0005, 'regularizer (weight_decay)')
flags.DEFINE_float('max_grad_clip', 0.25, 'max_grad_clip')
flags.DEFINE_float('max_grad_norm', 0, 'max_grad_norm')
flags.DEFINE_boolean('normalize', False, 'If set, zero-mean unit-variance normalization is used on data')
flags.DEFINE_boolean('last_only', True, 'save only the last checkpoint')
flags.DEFINE_string('load_checkpoint', '', 'name of the latest checkpoint to load')
flags.DEFINE_boolean('dynamic_train_split', True, 'Train samples are extracted on-the-fly. multiple seq samples are generated based on the window and stride sizes in preprocess_datasets.py')
flags.DEFINE_boolean('dynamic_val_split', False, 'Validation samples are extracted on-the-fly. multiple seq samples are generated based on the window and stride sizes in preprocess_datasets.py')

# EVALUATION
flags.DEFINE_boolean('dynamic_test_split', True, 'Test samples are extracted on-the-fly. multiple seq samples are generated based on the window and stride sizes in preprocess_datasets.py')
flags.DEFINE_boolean('visualize', False, 'Visualize ground-truth and predictions side-by-side by using human skeleton.')
flags.DEFINE_boolean('interactive', False, 'True, if motion is to be shown in an interactive matplotlib window.')
flags.DEFINE_boolean('to_video', True, 'Save the model predictions to mp4 videos in the experiments folder.')
flags.DEFINE_boolean('save_figs', True, 'Save the model predictions to jpeg images in the experiments folder.')
flags.DEFINE_boolean('dense', False, 'Show mesh in offline visualization')
flags.DEFINE_boolean('skel', True, 'Show skeleton in offline visualization')  
flags.DEFINE_boolean('one_batch', False, 'evaluation on only one batch')  
flags.DEFINE_multi_integer('METRIC_TARGET_LENGTHS', [1, 2, 4, 8, 10, 14, 25, 30],'# @ 25 Hz, in ms: [40, 80, 160, 320, 400, 560, 1000, 1200]')    

# GCN params
flags.DEFINE_integer('encoder_hidden', 256, 'default: 256')
flags.DEFINE_integer('encoder_mlp_hidden', 256, 'default: 256')
flags.DEFINE_integer('gcn_hidden', 256, 'default: 256')
flags.DEFINE_integer('num_edge_types', 2, 'default: 2')
flags.DEFINE_string('graph_type', 'dynamic', 'default: dynamic') 
flags.DEFINE_float('encoder_dropout', 0.3, 'default: 0.3')
flags.DEFINE_integer('encoder_mlp_num_layers', 3, 'default: 3')
flags.DEFINE_float('gumbel_temp', 0.5, 'default: 0.5')
flags.DEFINE_boolean('normalize_kl', True, 'default: True')
flags.DEFINE_float('kl_coef', 1.0, 'default: 1.0')
flags.DEFINE_float('no_edge_prior', 0.9, 'default: 0.9')
flags.DEFINE_float('mscale_dropout', 0.3, 'default: 0.3')
