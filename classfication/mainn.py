import numpy as np
from datetime import datetime
import argparse
from utils.utils import _logger
from model import Model
# Updated imports for new dataloader functions
from dataloaderr import (data_generator_pretraining_only, data_generator_finetuning_only, 
                       data_generator_combined, data_generator_only_ft)
from trainer import Trainer, linear_probe_eval
import os
import torch

# Args selections 
start_time = datetime.now()
parser = argparse.ArgumentParser()

home_dir = os.getcwd()
parser.add_argument('--run_description', default='run1', type=str, help='Experiment Description')
parser.add_argument('--seed', default=2025, type=int, help='seed value')

# Training modes: pre_train (TDMS only), fine_tune (Danfoss only), combined (TDMS pre-train + Danfoss fine-tune), linear_probe
parser.add_argument('--training_mode', default='fine_tune', type=str,
                    help='pre_train (TDMS only), fine_tune (Danfoss only), combined (TDMS pre-train + Danfoss fine-tune), linear_probe')

# Updated dataset arguments
parser.add_argument('--pretrain_dataset', default='TDMS_PowerData', type=str,
                    help='Pre-training dataset name for logging')
parser.add_argument('--target_dataset', default='Danfoss', type=str,
                    help='Fine-tuning dataset name for logging')

parser.add_argument('--logs_save_dir', default='/root/autodl-tmp', type=str, help='saving directory')
parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
parser.add_argument('--gpu_id', default=0, type=int, help='gpu_id')
parser.add_argument('--home_path', default=home_dir, type=str, help='Project home directory')
parser.add_argument('--subset', action='store_true', default=False, help='use the subset of datasets')
parser.add_argument('--log_epoch', default=5, type=int, help='print loss and metrix')
parser.add_argument('--draw_similar_matrix', default=10, type=int, help='draw similarity matrix')
parser.add_argument('--pretrain_lr', default=0.0001, type=float, help='pretrain learning rate')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--use_pretrain_epoch_dir', default=None, type=str,
                    help='choose the pretrain checkpoint to linear probe')
parser.add_argument('--pretrain_epoch', default=20, type=int, help='pretrain epochs')
parser.add_argument('--finetune_epoch', default=500, type=int, help='finetune epochs')

# TDMS and Danfoss specific arguments
parser.add_argument('--tdms_source_path', default='/root/autodl-tmp', type=str, help='Base path for TDMS source data')
parser.add_argument('--tdms_target_path', default='/root/autodl-tmp', type=str, help='Base path for TDMS target data')
parser.add_argument('--source_train_ratio', default=1, type=float, help='Training data ratio for TDMS pre-training')
parser.add_argument('--danfoss_train_ratio', default=1, type=float, help='Training data ratio for Danfoss fine-tuning')
parser.add_argument('--force_rebuild', action='store_true', default=False,
                    help='Force rebuild dataset cache even if it exists')

# Power level arguments (for TDMS data only)
parser.add_argument('--source_power_levels', default='1.0,1.5,3.0', type=str,
                    help='Comma-separated source power levels for pre-training (e.g., "1.0,1.5,3.0")')
parser.add_argument('--target_power_levels', default='1.5,3.0', type=str,
                    help='Comma-separated target power levels for TDMS fine-tuning (e.g., "1.5,3.0")')

# Model hyperparameters
parser.add_argument('--masking_ratio', default=0.5, type=float, help='masking ratio')
parser.add_argument('--positive_nums', default=3, type=int, help='positive series numbers')
parser.add_argument('--lm', default=3, type=int, help='average masked lenght')
parser.add_argument('--temperature', type=float, default=0.2, help='temperature')
parser.add_argument("--onelayer_out_channels", type=int, default=256, help="Conv1D_output_channels")
parser.add_argument("--twolayer_out_channels", type=int, default=512, help="Conv1D_output_channels")
parser.add_argument("--final_out_channels", type=int, default=1024, help="Conv1D_output_channels")
parser.add_argument("--kernel_size", type=int, default=25, help="kernal_size")
parser.add_argument("--exp_name", type=str, default="experiment", help="experiment name for exp record")

parser.add_argument("--conv2_kernel_size", type=int, default=8, help="conv2_kernel_size")
parser.add_argument("--conv3_kernel_size", type=int, default=8, help="conv3_kernel_size")
parser.add_argument("--d_ff", type=int, default=1024, help="hidden_size")
parser.add_argument("--CNNoutput_channel", type=int, default=4, help="CNNoutput_channel")
parser.add_argument("--ft_dropout", type=float, default=0, help="finetune_dropout")
parser.add_argument("--num_classes_target", type=int, default=5, help="target dataset class number (Danfoss: 5)")
parser.add_argument("--hidden_dimension", type=int, default=23552, help="hidden dimension")
parser.add_argument("--n_knlg", type=int, default=64, help="number of knowledge")
parser.add_argument("--stride", type=int, default=3, help="stride")

# Legacy arguments
parser.add_argument('--finetune_result_file_name', default="finetune_result.json", type=str,
                    help='finetune result json name')

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
    model = Model(configs, args)
    total_params = sum(p.numel() for p in model.parameters())
    # print(f"⚠️ 当前模型参数量: {total_params:,} ({total_params/1e6:.1f}M)")
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

    # Parse power levels (only used for TDMS data)
    source_power_levels = parse_power_levels(args.source_power_levels)
    target_power_levels = parse_power_levels(args.target_power_levels)
    
    # Set number of classes based on training mode
    if training_mode in ['fine_tune', 'linear_probe'] and targetdata == 'Danfoss':
        # Danfoss dataset has 5 fault levels (0, 0v41, 0v83, 1v25, 1v66)
        configs.num_classes_target = 5
        print(f"Setting num_classes_target to 5 for Danfoss dataset")
    elif training_mode in ['fine_tune', 'linear_probe']:
        # TDMS target data
        num_target_classes = len(target_power_levels) * 15
        configs.num_classes_target = num_target_classes
        print(f"Setting num_classes_target to {num_target_classes} "
              f"({len(target_power_levels)} power levels × 15 fault types)")
    else:
        # For pre-training or combined mode, use default
        configs.num_classes_target = args.num_classes_target
    
    # Update configs with model parameters
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
    configs.n_knlg = args.n_knlg
    configs.stride = args.stride
    
    # Update configs with data paths
    configs.tdms_source_path = args.tdms_source_path
    configs.tdms_target_path = args.tdms_target_path

    exp_name = args.exp_name
    os.makedirs(logs_save_dir, exist_ok=True)

    # Set seed
    if seed is not None:
        seed = set_seed(seed)
    else:
        seed = set_seed(args.seed)

    # Set device
    device = torch.device(f"cuda:{args.gpu_id}" if args.device == 'cuda' else 'cpu')
    
    # Create experiment description and directories
    experiment_description = f"{sourcedata}_2_{targetdata}"
    source_power_str = "_".join(source_power_levels) if training_mode != 'fine_tune' else "none"
    target_power_str = "_".join(target_power_levels) if training_mode != 'fine_tune' else "danfoss"
    
    experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description,
                                      f"{training_mode}_{source_power_str}_to_{target_power_str}_" \
                                      f"{seed}_pt_{masking_ratio}_{pretrain_lr}_{pretrain_epoch}_" \
                                      f"ft_{lr}_{finetune_epoch}_expname_{exp_name}")
    os.makedirs(experiment_log_dir, exist_ok=True)

    # Setup logging
    log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
    logger = _logger(log_file_name)
    
    # Data loading based on training mode
    logger.debug("=" * 45)
    logger.debug(f'Training Mode: {training_mode}')
    logger.debug(f'Pre-training Dataset: {sourcedata}')
    logger.debug(f'Target Dataset: {targetdata}')
    
    train_dl, valid_dl, test_dl = None, None, None

    if training_mode == 'pre_train':
        # Pre-training only with TDMS data (all power levels)
        logger.debug("Loading TDMS data for pre-training only...")
        logger.debug(f'Source Power Levels: {source_power_levels}')
        logger.debug(f'Source Train Ratio: {args.source_train_ratio}')
        
        train_dl = data_generator_pretraining_only(
            sourcedata_path=args.tdms_source_path,
            configs=configs,
            training_mode=training_mode,
            subset=args.subset,
            source_train_ratio=args.source_train_ratio,
            force_rebuild=args.force_rebuild
        )
        valid_dl = None  # No validation for pre-training only
        test_dl = None   # No test for pre-training only
        
    elif training_mode == 'fine_tune':
        # ✅ Fine-tuning only with Danfoss data
        logger.debug("Loading Danfoss data for fine-tuning only...")
        logger.debug(f'Danfoss Train Ratio: {args.danfoss_train_ratio}')
        
        finetune_loader, test_loader = data_generator_finetuning_only(
            configs=configs,
            training_mode=training_mode,
            subset=args.subset,
            danfoss_train_ratio=args.danfoss_train_ratio,
            force_rebuild=args.force_rebuild
        )
        
        train_dl = finetune_loader
        valid_dl = None
        test_dl = test_loader

    elif training_mode == 'combined':
        # Combined: TDMS pre-training + Danfoss fine-tuning
        logger.debug("Loading combined data: TDMS pre-training + Danfoss fine-tuning...")
        logger.debug(f'Source Power Levels: {source_power_levels}')
        logger.debug(f'Source Train Ratio: {args.source_train_ratio}')
        logger.debug(f'Danfoss Train Ratio: {args.danfoss_train_ratio}')
        
        train_dl, valid_dl, test_dl = data_generator_combined(
            sourcedata_path=args.tdms_source_path,
            configs=configs,
            training_mode=training_mode,
            subset=args.subset,
            source_train_ratio=args.source_train_ratio,
            danfoss_train_ratio=args.danfoss_train_ratio,
            force_rebuild=args.force_rebuild
        )
        
    elif training_mode == 'linear_probe':
        # Linear probe evaluation
        logger.debug("Linear Probe evaluation started ....")
        
        if args.use_pretrain_epoch_dir:
            chkpoint = torch.load(args.use_pretrain_epoch_dir)
            logger.debug(f"Loaded pretrained model from {args.use_pretrain_epoch_dir}")
        else:
            last_epoch = pretrain_epoch
            chkpoint_path = os.path.join(logs_save_dir, experiment_description, run_description,
                                      f"combined_{source_power_str}_to_{target_power_str}_" \
                                      f"{seed}_pt_{masking_ratio}_{pretrain_lr}_{pretrain_epoch}_" \
                                      f"ft_{lr}_{finetune_epoch}_expname_{exp_name}",
                                      "saved_models", f"ckp_ep{last_epoch}.pt")
            if os.path.exists(chkpoint_path):
                chkpoint = torch.load(chkpoint_path)
                logger.debug(f"Loaded pretrained model from {chkpoint_path}")
            else:
                logger.debug(f"Checkpoint not found: {chkpoint_path}")
                raise FileNotFoundError(f"Pretrained model not found at {chkpoint_path}")
        
        model = Model(configs, args).to(device)
        model.load_state_dict(chkpoint["model_state_dict"])
        
        train_dl_probe, test_dl = data_generator_finetuning_only(
            configs=configs,
            training_mode='fine_tune',
            subset=args.subset,
            danfoss_train_ratio=0.8,
            force_rebuild=args.force_rebuild
        )
        
        test_loss, test_acc, test_auc, test_prc, performance = linear_probe_eval(
            model, train_dl_probe, test_dl, device, configs
        )
        
        logger.debug(f"Linear Probe Test Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")
        logger.debug(
            'Linear Probe - Testing: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f' % (
            performance[0], performance[1], performance[2], performance[3]))
        
        log_file_name = os.path.join(experiment_log_dir, f"linear_probe_results.txt")
        with open(log_file_name, "w") as f:
            f.write(f"Linear Probe Test Loss: {test_loss:.4f}\n")
            f.write(f"Accuracy: {performance[0]:.4f}\n")
            f.write(f"Precision: {performance[1]:.4f}\n")
            f.write(f"Recall: {performance[2]:.4f}\n")
            f.write(f"F1: {performance[3]:.4f}\n")
        
        logger.debug(f"Linear Probe results saved to {log_file_name}")
        return performance
    
    else:
        raise ValueError(f"Unknown training mode: {training_mode}")
    
    # Log configuration
    logger.debug(f'Target Number of Classes: {configs.num_classes_target}')
    logger.debug(f'Seed: {seed}')
    logger.debug(f'Method: {method}')
    logger.debug(f'Pretrain Learning rate: {pretrain_lr}')
    logger.debug(f'Masking ratio: {masking_ratio}')
    logger.debug(f'Pretrain Epochs: {pretrain_epoch}')
    logger.debug(f'Finetune Learning rate: {lr}')
    logger.debug(f'Finetune Epochs: {finetune_epoch}')
    logger.debug(f'Temperature: {temperature}')
    logger.debug(f"Input channels: {configs.input_channels}")
    logger.debug(f"onelayer_out_channels: {configs.onelayer_out_channels}")
    logger.debug(f"twolayer_out_channels: {configs.twolayer_out_channels}")
    logger.debug(f"final_out_channels: {configs.final_out_channels}")
    logger.debug(f"kernel_size: {configs.kernel_size}")
    logger.debug(f"exp_name: {exp_name}")
    logger.debug(f"ft_dropout: {configs.ft_dropout}")
    logger.debug(f"n_knlg: {configs.n_knlg}")
    logger.debug(f"force_rebuild: {args.force_rebuild}")
    
    if training_mode != 'fine_tune':
        logger.debug(f"source_train_ratio: {args.source_train_ratio}")
    if training_mode != 'pre_train':
        logger.debug(f"danfoss_train_ratio: {args.danfoss_train_ratio}")
    
    logger.debug("=" * 45)

    # Load Model
    model = Model(configs, args).to(device)

    if training_mode == 'fine_tune':
        if args.use_pretrain_epoch_dir:
            chkpoint = torch.load(args.use_pretrain_epoch_dir)
            model.load_state_dict(chkpoint["model_state_dict"])
            print("\n" + "="*50)
            print("🔍 PRETRAINED MODEL STRUCTURE:")
            print("="*50)
            for name in chkpoint["model_state_dict"].keys():
                if "channel_attn" in name:
                    print(f"  {name}")
            logger.debug(f"✅ Loaded pretrained model from: {args.use_pretrain_epoch_dir}")
        else:
            pretrain_experiment_log_dir = os.path.join(
                args.logs_save_dir, experiment_description, args.run_description,
                f"combined_none_to_danfoss_{args.seed}_pt_{args.masking_ratio}_{args.pretrain_lr}_{args.pretrain_epoch}_"
                f"ft_{args.lr}_{args.finetune_epoch}_expname_{args.exp_name}"
            )
            chkpoint_path = os.path.join(pretrain_experiment_log_dir, "saved_models", f"ckp_ep{args.pretrain_epoch}.pt")
            
            if os.path.exists(chkpoint_path):
                chkpoint = torch.load(chkpoint_path)
                model.load_state_dict(chkpoint["model_state_dict"])
                logger.debug(f"✅ Auto-loaded pretrained model from: {chkpoint_path}")
            else:
                logger.debug(f"⚠️ No pretrained model found at {chkpoint_path}. Using random initialization.")

    # Optimizer
    params_group = [{'params': model.parameters()}]
    if training_mode == 'fine_tune':
        model_optimizer = torch.optim.Adam(params_group, lr=lr, betas=(configs.beta1, configs.beta2),
                                           weight_decay=0)
        model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optimizer, T_max=finetune_epoch)
    else:
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
    
    # Load configuration
    try:
        exec(f'from config_files.Koera_Configs import Config as Configs')
        configs = Configs()
    except ImportError:
        print(f"Config file not found, using default config")
        try:
            exec(f'from config_files.default_Config import Config as Configs')
            configs = Configs()
        except ImportError:
            class DefaultConfig:
                def __init__(self):
                    self.batch_size = 32
                    self.target_batch_size = 16
                    self.drop_last = True
                    self.TSlength_aligned = 512
                    
                    if args.training_mode == 'fine_tune' and args.target_dataset == 'Danfoss':
                        self.input_channels = 10
                        self.sequence_len = 512
                    else:
                        self.input_channels = 6
                        self.sequence_len = 512
                    
                    self.beta1 = 0.9
                    self.beta2 = 0.999
                    
                    self.onelayer_out_channels = 64
                    self.twolayer_out_channels = 128
                    self.final_out_channels = 256
                    self.kernel_size = 25
                    self.conv2_kernel_size = 8
                    self.conv3_kernel_size = 8
                    self.d_ff = 256
                    self.CNNoutput_channel = 4
                    self.ft_dropout = 0
                    self.hidden_dimension = 5888
                    self.n_knlg = 16
                    self.stride = 3
                    self.num_classes_target = 5
            
            configs = DefaultConfig()
            print("Using default configuration")
    
    main(args, configs)
