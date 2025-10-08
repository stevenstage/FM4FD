import numpy as np
from datetime import datetime
import argparse
from utils.utils import _logger
from model import Model
from dataloader import data_generator, data_generator_only_ft
from trainer import Trainer, linear_probe_eval
import os
import torch

# Args selections
start_time = datetime.now()
parser = argparse.ArgumentParser()

home_dir = os.getcwd()
parser.add_argument('--run_description', default='run1', type=str, help='Experiment Description')
parser.add_argument('--seed', default=2025, type=int, help='seed value')

parser.add_argument('--training_mode', default='pre_train', type=str, help='pre_train, fine_tune, linear_probe')
# Modified: Changed default datasets to reflect your TDMS data
parser.add_argument('--pretrain_dataset', default='3.0kW', type=str,
                    help='Dataset of choice: PowerData_Source, SleepEEG, FD_A, HAR, ECG')
parser.add_argument('--target_dataset', default='1.0kW+1.5KW', type=str,
                    help='Dataset of choice: PowerData_Target, Epilepsy, FD_B, Gesture, EMG')

parser.add_argument('--logs_save_dir', default='experiments_FAT_logs', type=str, help='saving directory')
parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
parser.add_argument('--gpu_id',default=0,type=int,help='gpu_id')
parser.add_argument('--home_path', default=home_dir, type=str, help='Project home directory')
parser.add_argument('--subset', action='store_true', default=False, help='use the subset of datasets')
parser.add_argument('--log_epoch', default=5, type=int, help='print loss and metrix')
parser.add_argument('--draw_similar_matrix', default=10, type=int, help='draw similarity matrix')
parser.add_argument('--pretrain_lr', default=0.0001, type=float, help='pretrain learning rate')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--use_pretrain_epoch_dir', default=None, type=str,
                    help='choose the pretrain checkpoint to linear probe')
parser.add_argument('--pretrain_epoch', default=10, type=int, help='pretrain epochs')
parser.add_argument('--finetune_epoch', default=60, type=int, help='finetune epochs')

# Added: New arguments specific to your TDMS data
parser.add_argument('--train_ratio', default=0.01, type=float, help='Training data ratio for TDMS processing')
parser.add_argument('--tdms_data_path', default='/root/autodl-tmp', type=str, help='Base path for TDMS data')
parser.add_argument('--use_combined_data', action='store_true', default=True, 
                    help='Use combined current and vibration data')
parser.add_argument('--force_rebuild', action='store_true', default=False,
                    help='Force rebuild dataset cache even if it exists')
parser.add_argument('--source_power_levels', default='3.0', type=str,
                    help='Comma-separated source power levels for pre-training (e.g., "1.0,1.5")')
parser.add_argument('--target_power_levels', default='1.0,1.5', type=str,
                    help='Comma-separated target power levels for linear probe (e.g., "1.5,3.0")')

parser.add_argument('--masking_ratio', default=0.5, type=float, help='masking ratio')
parser.add_argument('--positive_nums', default=3, type=int, help='positive series numbers')
parser.add_argument('--lm', default=3, type=int, help='average masked lenght')

parser.add_argument('--finetune_result_file_name', default="finetune_result.json", type=str,
                    help='finetune result json name')
parser.add_argument('--temperature', type=float, default=0.2, help='temperature')
parser.add_argument("--onelayer_out_channels",type=int,default=64,help="Conv1D_output_channels")
parser.add_argument("--twolayer_out_channels",type=int,default=128,help="Conv1D_output_channels")
parser.add_argument("--final_out_channels",type=int,default=256,help="Conv1D_output_channels")
parser.add_argument("--kernel_size",type=int,default=25,help="kernal_size")
parser.add_argument("--exp_name",type=str,default="experiment",help="experiment name for exp record")
parser.add_argument("--conv2_kernel_size",type=int,default=8,help="conv2_kernel_size")
parser.add_argument("--conv3_kernel_size",type=int,default=8,help="conv3_kernel_size")
parser.add_argument("--d_ff",type=int,default=256,help="hidden_size")
parser.add_argument("--CNNoutput_channel",type=int,default=4,help="CNNoutput_channel")
parser.add_argument("--ft_dropout",type=float,default=0,help="finetune_dropout")
parser.add_argument("--num_classes_target",type=int,default=30,help="target dataset class number")
parser.add_argument("--hidden_dimension",type=int,default=2560,help="hidden dimension")
parser.add_argument("--n_knlg",type=int,default=16,help="number of knowledge")
parser.add_argument("--stride",type=int,default=3,help="stride")

def set_seed(seed):
    SEED = seed
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

def parse_power_levels(power_str):
    levels = []
    for level in power_str.split(','):
        level = level.strip()
        if level.endswith('kW'):
            level = level[:-2]
        levels.append(level)
    return levels

def main(args, configs, seed=None):
    method = 'SimMTM'
    sourcedata = args.pretrain_dataset
    targetdata = args.target_dataset
    training_mode = args.training_mode
    run_description = args.run_description

    logs_save_dir = args.logs_save_dir
    masking_ratio = args.masking_ratio
    pretrain_lr = args.pretrain_lr
    pretrain_epoch = args.pretrain_epoch
    lr = args.lr
    finetune_epoch = args.finetune_epoch
    temperature = args.temperature
    experiment_description = f"{sourcedata}_2_{targetdata}"

    source_power_levels = parse_power_levels(args.source_power_levels)
    target_power_levels = parse_power_levels(args.target_power_levels)
    
    num_target_classes = len(target_power_levels) * 15
    configs.num_classes_target = num_target_classes
    print(f"Setting num_classes_target to {num_target_classes} "
          f"({len(target_power_levels)} power levels × 15 fault types)")
    
    configs.onelayer_out_channels = args.onelayer_out_channels
    configs.twolayer_out_channels = args.twolayer_out_channels
    configs.final_out_channels = args.final_out_channels
    configs.kernel_size = args.kernel_size
    configs.conv2_kernel_size = args.conv2_kernel_size
    configs.conv3_kernel_size = args.conv3_kernel_size
    configs.d_ff = args.d_ff
    configs.CNNoutput_channel = args.CNNoutput_channel
    configs.ft_dropout = args.ft_dropout
    configs.hidden_dimension = args.hidden_dimension
    exp_name = args.exp_name
    configs.n_knlg = args.n_knlg
    configs.stride = args.stride

    os.makedirs(logs_save_dir, exist_ok=True)

    # Modified: Load datasets - changed paths to work with your TDMS data structure
    sourcedata_path = f"/root/autodl-tmp/{sourcedata}"
    targetdata_path = f"/root/autodl-tmp/{targetdata}"
    
    # Create directories if they don't exist
    os.makedirs(sourcedata_path, exist_ok=True)
    os.makedirs(targetdata_path, exist_ok=True)

    subset = args.subset  # if subset= true, use a subset for debugging.
    
    device = torch.device(f"cuda:{args.gpu_id}" if args.device == 'cuda' else 'cpu')
    
    if training_mode == 'linear_probe':
        log_file_name = f"logs_linear_probe_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
        logger = _logger(log_file_name)
        logger.debug("Linear Probe evaluation started ....")
        
        if args.use_pretrain_epoch_dir:
            chkpoint = torch.load(args.use_pretrain_epoch_dir)
            logger.debug(f"Loaded pretrained model from {args.use_pretrain_epoch_dir}")
        else:
            last_epoch = pretrain_epoch
            chkpoint_path = os.path.join(logs_save_dir, experiment_description, run_description,
                                      f"pre_train_{seed}_pt_{masking_ratio}_{pretrain_lr}_{pretrain_epoch}_ft_{lr}_{finetune_epoch}_expname_{exp_name}",
                                      "saved_models", f"ckp_ep{last_epoch}.pt")
            chkpoint = torch.load(chkpoint_path)
            logger.debug(f"Loaded pretrained model from {chkpoint_path}")
        
        model = Model(configs, args).to(device)
        model.load_state_dict(chkpoint["model_state_dict"])
        
        train_dl, _, test_dl = data_generator_only_ft(
            targetdata_path, configs, 'linear_probe', 
            power_levels=target_power_levels,
            subset=subset,
            train_ratio=0.01,
            force_rebuild=args.force_rebuild
        )
        
        test_loss, test_acc, test_auc, test_prc, performance = linear_probe_eval(
            model, train_dl, test_dl, device, configs
        )
        
        logger.debug(f"Linear Probe Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        logger.debug(
            'Linear Probe - Testing: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f' % (
            performance[0], performance[1], performance[2], performance[3]))
        
        experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description,
                                      f"linear_probe_{args.source_power_levels}_to_{args.target_power_levels}_" \
                                      f"{seed}_pt_{masking_ratio}_{pretrain_lr}_{pretrain_epoch}_" \
                                      f"ft_{lr}_{finetune_epoch}_expname_{exp_name}")
        os.makedirs(experiment_log_dir, exist_ok=True)
        
        log_file_name = os.path.join(experiment_log_dir, f"linear_probe_results.txt")
        with open(log_file_name, "w") as f:
            f.write(f"Linear Probe Test Loss: {test_loss:.4f}\n")
            f.write(f"Accuracy: {performance[0]:.4f}\n")
            f.write(f"Precision: {performance[1]:.4f}\n")
            f.write(f"Recall: {performance[2]:.4f}\n")
            f.write(f"F1: {performance[3]:.4f}\n")
        
        logger.debug(f"Linear Probe results saved to {log_file_name}")
        
        return performance
        
    elif training_mode == 'fine_tune' or training_mode == 'ft':
        # Use only target data for fine-tuning
        train_dl, valid_dl, test_dl = data_generator_only_ft(
            targetdata_path, configs, training_mode, 
            power_levels=target_power_levels,
            subset=subset,
            train_ratio=args.train_ratio, 
            force_rebuild=args.force_rebuild
        )
        logger_info = f"Using fine-tune only mode with target data from {targetdata_path}, " \
                      f"power_levels={target_power_levels}, train_ratio={args.train_ratio}"
    else:
        # Use both source and target data for pre-training + fine-tuning
        # For source dataset (pre-training), use higher ratio
        source_train_ratio = 1  # Scale up for pre-training
        train_dl, valid_dl, test_dl = data_generator(
            sourcedata_path, targetdata_path, configs, training_mode, 
            source_power_levels=source_power_levels,
            target_power_levels=target_power_levels,
            subset=subset, 
            source_train_ratio=source_train_ratio, 
            target_train_ratio=args.train_ratio,
            force_rebuild=args.force_rebuild
        )
        logger_info = f"Using pre-train mode with source: {sourcedata_path} (power_levels={source_power_levels}, train_ratio={source_train_ratio}), " \
                      f"target: {targetdata_path} (power_levels={target_power_levels}, train_ratio={args.train_ratio})"
    
    # set seed
    if seed is not None:
        seed = set_seed(seed)
    else:
        seed = set_seed(args.seed)

    source_power_str = "_".join(source_power_levels)
    target_power_str = "_".join(target_power_levels)
    experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description,
                                      f"{training_mode}_{source_power_str}_to_{target_power_str}_" \
                                      f"{seed}_pt_{masking_ratio}_{pretrain_lr}_{pretrain_epoch}_" \
                                      f"ft_{lr}_{finetune_epoch}_expname_{exp_name}")
    os.makedirs(experiment_log_dir, exist_ok=True)

    # Logging
    log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    logger.debug("=" * 45)
    logger.debug(f'Pre-training Dataset: {sourcedata}')
    logger.debug(f'Target (fine-tuning) Dataset: {targetdata}')
    logger.debug(f'Source Power Levels: {source_power_levels}')
    logger.debug(f'Target Power Levels: {target_power_levels}')
    logger.debug(f'Target Number of Classes: {num_target_classes} ({len(target_power_levels)} × 15)')
    if training_mode == 'fine_tune' or training_mode == 'ft':
        logger.debug(logger_info)  # Added: Log data loading info
    logger.debug(f'Seed: {seed}')
    logger.debug(f'Method:  {method}')
    logger.debug(f'Mode:    {training_mode}')
    logger.debug(f'Pretrain Learning rate:    {pretrain_lr}')
    logger.debug(f'Masking ratio:    {masking_ratio}')
    logger.debug(f'Pretrain Epochs:    {pretrain_epoch}')
    logger.debug(f'Finetune Learning rate:    {lr}')
    logger.debug(f'Finetune Epochs:    {finetune_epoch}')
    logger.debug(f'Temperature: {temperature}')
    logger.debug(f"Input channels: {configs.input_channels}")
    logger.debug(f"onelayer_out_channels:{configs.onelayer_out_channels}")
    logger.debug(f"twolayer_out_channels:{configs.twolayer_out_channels}")
    logger.debug(f"final_out_channels:{configs.final_out_channels}")
    logger.debug(f"kernel_size:{configs.kernel_size}")
    logger.debug(f"exp_name:{exp_name}")
    logger.debug(f"ft_dropout:{configs.ft_dropout}")
    logger.debug(f"n_knlg:{configs.n_knlg}")
    logger.debug(f"num_classes_target:{configs.num_classes_target}")
    logger.debug(f"train_ratio:{args.train_ratio}")
    logger.debug(f"force_rebuild:{args.force_rebuild}")

    logger.debug("=" * 45)

    # Load Model
    model = Model(configs, args).to(device)
    params_group = [{'params': model.parameters()}]
    model_optimizer = torch.optim.Adam(params_group, lr=pretrain_lr, betas=(configs.beta1, configs.beta2),
                                       weight_decay=0)
    model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optimizer, T_max=pretrain_epoch)

    # Trainer
    best_performance = Trainer(model, model_optimizer, model_scheduler, train_dl, valid_dl, test_dl, device, logger,
                               args, configs, experiment_log_dir, seed)

    return best_performance


if __name__ == '__main__':
    args, unknown = parser.parse_known_args()
    device = torch.device(f"cuda:{args.gpu_id}" if args.device == 'cuda' else 'cpu')
    
    # Modified: Create a custom config for your TDMS data if needed, or use existing config
    try:
        # Try to use existing config structure
        exec (f'from config_files.SleepEEG_Configs import Config as Configs')
        configs = Configs()
    except ImportError:
        # If no specific config exists, create a basic one or use a default
        print(f"Config file for {args.pretrain_dataset} not found, using default config")
        try:
            exec (f'from config_files.Koera_Configs import Config as Configs')
            configs = Configs()
        except ImportError:
            # Create a minimal config class if no config files exist
            class DefaultConfig:
                def __init__(self):
                    self.batch_size = 32
                    self.target_batch_size = 16
                    self.drop_last = True
                    self.TSlength_aligned = 512
                    self.input_channels = 6
                    self.sequence_len = 512
                    self.beta1 = 0.9
                    self.beta2 = 0.999
            
            configs = DefaultConfig()
    
    main(args, configs)
