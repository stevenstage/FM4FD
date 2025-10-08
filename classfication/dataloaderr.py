import torch
from torch.utils.data import DataLoader, Dataset, random_split
import os
import numpy as np
from nptdms import TdmsFile
import json
from datetime import datetime
import scipy.io
from typing import Tuple

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode, power_levels, target_dataset_size=64, subset=False):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode
        self.power_levels = power_levels
        X_train = dataset["samples"]
        y_train = dataset["labels"]
        
        # Create power level mapping for TDMS data
        self.power_mapping = {'1.0': 0, '1.5': 15, '3.0': 30}
        
        # Calculate offset for each power level in target domain
        self.offsets = {}
        base = 0
        for power_level in sorted(power_levels):
            self.offsets[power_level] = base
            base += 15  # Each power level adds 15 fault types
        
        # shuffle
        data = list(zip(X_train, y_train))
        np.random.shuffle(data)
        X_train, y_train = zip(*data)
        X_train, y_train = torch.stack(list(X_train), dim=0), torch.stack(list(y_train), dim=0)

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        """Align the TS length between source and target datasets"""
        X_train = X_train[:, :, :int(config.TSlength_aligned)]

        """Subset for debugging"""
        if subset == True:
            subset_size = target_dataset_size * 10
            X_train = X_train[:subset_size] 
            y_train = y_train[:subset_size]

        # Apply dynamic label mapping for TDMS data only
        if hasattr(self, 'power_mapping') and any(str(pl) in self.power_mapping for pl in power_levels):
            y_train = self.map_labels(y_train)
        
        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]

    def map_labels(self, labels):
        """Map original labels to continuous range based on target power levels"""
        new_labels = []
        for label in labels:
            label = label.item() if torch.is_tensor(label) else label
            
            # Find which power level this label belongs to
            for power_level, start_label in self.power_mapping.items():
                if start_label <= label < start_label + 15:
                    # Calculate fault type within the power level
                    fault_type = label - start_label
                    # Apply offset based on target domain
                    if power_level in self.offsets:
                        new_label = self.offsets[power_level] + fault_type
                        new_labels.append(new_label)
                    else:
                        # This should not happen - label from power level not in target domain
                        raise ValueError(f"Label {label} from power level {power_level} not in target domain {self.power_levels}")
                    break
            else:
                raise ValueError(f"Label {label} doesn't match any power level range")
        
        return np.array(new_labels)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


# ============= TDMS Data Processing Functions =============
def get_tdms_files(directory):
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.tdms')])

def read_tdms_file(filepath):
    tdms_file = TdmsFile.read(filepath)
    data_list = []

    # Extract data from all channels
    for group in tdms_file.groups():
        for channel in group.channels():
            data_list.append(channel[:])

    data = np.stack(data_list, axis=0)
    max_length = 12000000 if data.shape[0] == 3 else 3072000
    return data[:, :max_length]

def read_all_tdms_files(directory, start_label):
    data_list, label_list = [], []
    files = get_tdms_files(directory)
    
    if len(files) >= 2:
        data1 = read_tdms_file(files[0])
        data2 = read_tdms_file(files[1])
        combined_data = np.concatenate([data1, data2], axis=1)
        data_list.append(combined_data)
        label_list.append(start_label)
        
        for idx, file in enumerate(files[2:], start=1):
            data_list.append(read_tdms_file(file))
            label_list.append(start_label + idx)
    
    return data_list, label_list

def segment_tdms(data_list, label_list):
    segments, segment_labels = [], []
    
    for data, label in zip(data_list, label_list):
        window_size = 2000 if data.shape[0] == 3 else 512
        
        for i in range(0, data.shape[1] - window_size + 1, window_size):
            segments.append(data[:, i:i + window_size])
            segment_labels.append(label)
    
    segments = np.stack(segments, axis=0)
    segment_labels = np.stack(segment_labels, axis=0)
    
    idx = np.arange(len(segments))
    np.random.shuffle(idx)
    return segments[idx], segment_labels[idx]

def load_power_data(power_level, data_type='current', base_tdms_path='/root/autodl-tmp'):
    if power_level.endswith('kW'):
        # Handle direct kW specification like '1.0kW'
        level_num = power_level[:-2]  # Remove 'kW'
        base_path = f'{base_tdms_path}/{power_level}/{data_type}/{data_type}'
    else:
        # Handle numeric specification like '1.0'
        level_num = power_level
        base_path = f'{base_tdms_path}/{power_level}kW/{data_type}/{data_type}'
    
    power_mapping = {'1.0': 0, '1.5': 15, '3.0': 30}
    start_label = power_mapping[level_num]
    return read_all_tdms_files(base_path, start_label=start_label)

def prepare_single_power_data(power_level, base_tdms_path='/root/autodl-tmp', train_ratio=0.8):
    """
    Prepare data from a single power level (for source or target dataset)
    """
    print(f"Processing power level: {power_level} from path: {base_tdms_path}")
    
    # Load and process data for the specified power level
    c_data, labels = load_power_data(power_level, 'current', base_tdms_path)
    c_data, labels = segment_tdms(c_data, labels)
    print(f"Current data shape: {c_data.shape}, Labels: {len(labels)}")
    
    v_data, _ = load_power_data(power_level, 'vibration', base_tdms_path)
    v_data, _ = segment_tdms(v_data, _)
    print(f"Vibration data shape: {v_data.shape}")
    
    # Downsample current data to match vibration resolution
    indices = np.linspace(0, 1999, 512).astype(int)
    c_data = c_data[:, :, indices]
    print(f"Downsampled current data shape: {c_data.shape}")
    
    # Combine current and vibration data
    combined_data = np.concatenate((c_data, v_data), axis=1)
    print(f"Combined data shape: {combined_data.shape}")
    
    # Shuffle data
    idx = np.arange(len(combined_data))
    np.random.shuffle(idx)
    combined_data, labels = combined_data[idx], labels[idx]
    
    # Convert to tensors
    combined_data = torch.tensor(combined_data, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    
    # Split into train, validation, and test sets
    total_size = len(combined_data)
    test_size = int(total_size * 0.01)
    remaining_size = total_size - test_size
    train_size = int(remaining_size * train_ratio)
    val_size = remaining_size - train_size
    
    print(f"Data split - Total: {total_size}, Train: {train_size}, Val: {val_size}, Test: {test_size}")
    
    # Get indices for splits
    indices = np.random.permutation(total_size)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:test_size + train_size]
    val_indices = indices[test_size + train_size:]
    
    # Create datasets in official format
    train_dataset = {
        "samples": combined_data[train_indices],
        "labels": labels[train_indices]
    }
    
    val_dataset = {
        "samples": combined_data[val_indices],
        "labels": labels[val_indices]
    }
    
    test_dataset = {
        "samples": combined_data[test_indices],
        "labels": labels[test_indices]
    }
    
    return train_dataset, val_dataset, test_dataset


# ============= Danfoss Data Processing Functions =============
def combine_time_series(data_dict):
    shapes = [data.shape for data in data_dict.values()]
    unique_shapes = set(shapes)
    if len(unique_shapes) > 1:
        raise ValueError(f"Not all data arrays have the same shape. Shapes found: {unique_shapes}")
    
    data_arrays = list(data_dict.values())
    combined_data = np.stack(data_arrays, axis=-1)
    
    return combined_data

def load_data_and_labels(mat_file_paths, labels, operating_conditions):
    if len(mat_file_paths) != len(labels) or len(mat_file_paths) != len(operating_conditions):
        raise ValueError("Number of files must match the number of labels and operating conditions.")
    
    data_list = []
    labels_list = []
    op_cond_list = []
    variable_names = [
        'P3504_S101_U_phase_current_instantaneous_value',
        'P3505_S101_V_phase_current_instantaneous_value',
        'P3506_S101_W_phase_current_instantaneous_value',
        'S101_TorqueCurrentPct', 
        'S101_FluxCurrentPct',
        'S101_UL3L1Act',
        'S101_UL2L3Act',
        'S101_UL1L2Act',
        'P9009_S101_Motor_Torque',
        'P9015_S101_Output_Frequency'
    ]

    for file_path, label, op_cond in zip(mat_file_paths, labels, operating_conditions):
        try:
            mat_data = scipy.io.loadmat(file_path)
            data_dict = {}
            for name in variable_names:
                if name in mat_data:
                    data_dict[name] = mat_data[name]
                else:
                    raise KeyError(f"Missing variable '{name}' in {file_path}")
            
            combined_data = combine_time_series(data_dict)
            time_series_data = combined_data[:, 1, :]
            
            data_list.append(time_series_data)
            labels_list.append(label)
            op_cond_list.append(op_cond)
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    return data_list, labels_list, op_cond_list

def search_mat_files(folder_list):
    mat_files = []
    operating_conditions = []
    for folder in folder_list:
        op_cond = int(folder.split('_')[-1])
        
        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.mat'):
                    mat_files.append(os.path.join(root, file))
                    operating_conditions.append(op_cond)
    return mat_files, operating_conditions

def extract_fault_level_from_filename(filename):
    basename = os.path.basename(filename)
    parts = basename.split('_')
    last_part = parts[-1].split('.')[0]
    
    fault_level_mapping = {
        '0': 0,
        '0v41': 1,
        '0v83': 2,
        '1v25': 3,
        '1v66': 4
    }
    
    return fault_level_mapping.get(last_part, -1)

def segment_danfoss(data_list, label_list):
    split_data_list = []
    split_label_list = []

    for index, data in enumerate(data_list):
        step_size = 512
        move_size = 30

        for i in range(0, data.shape[0] - step_size + 1, move_size):
            split_data_list.append(data[i:i + step_size, :].T)
            split_label_list.append(label_list[index])

    if not split_data_list:
        return np.array([]), np.array([])
        
    split_data = np.stack(split_data_list, axis=0)
    split_label = np.array(split_label_list)
    
    return split_data, split_label

def load_danfoss_data(train_ratio=1.0):
    print(f"Loading Danfoss data with train_ratio: {train_ratio}")
    
    # All operating conditions are used for fine-tuning
    data_folder_list = [
        '/root/autodl-tmp/f_10',
        '/root/autodl-tmp/f_25',
        '/root/autodl-tmp/f_35',
        '/root/autodl-tmp/f_50'
    ]

    mat_file_paths, operating_conditions = search_mat_files(data_folder_list)
    print(f"Found {len(mat_file_paths)} .mat files in total.")
    
    fault_levels = [extract_fault_level_from_filename(file_path) for file_path in mat_file_paths]
    
    valid_indices = [i for i, label in enumerate(fault_levels) if label != -1]
    valid_file_paths = [mat_file_paths[i] for i in valid_indices]
    valid_fault_levels = [fault_levels[i] for i in valid_indices]
    valid_op_conds = [operating_conditions[i] for i in valid_indices]
    
    print(f"Valid files: {len(valid_file_paths)}")
    
    if not valid_file_paths:
        print("No valid files found. Exiting.")
        return (np.array([]), np.array([]))

    data_list, fault_levels_list, op_conds_list = load_data_and_labels(
        valid_file_paths, valid_fault_levels, valid_op_conds
    )
    
    print(f"Loaded {len(data_list)} files from all operating conditions")
    
    if 0 < train_ratio < 1 and len(data_list) > 0:
        indices = np.arange(len(data_list))
        np.random.shuffle(indices)
        split_idx = int(len(data_list) * train_ratio)
        selected_indices = indices[:split_idx]
        
        data_list = [data_list[i] for i in selected_indices]
        fault_levels_list = [fault_levels_list[i] for i in selected_indices]
        
        print(f"After applying train_ratio={train_ratio}, using {len(data_list)} files")
    
    X_data, y_data = segment_danfoss(data_list, fault_levels_list)
    
    print(f"Final Danfoss data shape: {X_data.shape if X_data.size > 0 else 0}")
    print(f"Final Danfoss labels shape: {y_data.shape if y_data.size > 0 else 0}")
    
    return X_data, y_data


# ============= Cache Management Functions =============
def clear_cache_if_params_changed(cache_path, current_params):
    meta_path = cache_path + "_meta.json"
    
    if not os.path.exists(meta_path):
        return False
    
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        param_match = True
        for key, value in current_params.items():
            if key not in meta['params'] or meta['params'][key] != value:
                param_match = False
                break
        
        if not param_match:
            for ext in ['.pt', '_meta.json']:
                if os.path.exists(cache_path + ext):
                    os.remove(cache_path + ext)
            print(f"Cleared cache due to parameter change: {current_params} != {meta['params']}")
            return True
        return False
    
    except Exception as e:
        print(f"Error reading meta {e}")
        for ext in ['.pt', '_meta.json']:
            if os.path.exists(cache_path + ext):
                os.remove(cache_path + ext)
        return True

def save_with_metadata(cache_path, dataset, params):
    torch.save(dataset, cache_path + ".pt")
    
    meta = {
        'timestamp': datetime.now().isoformat(),
        'params': params
    }
    with open(cache_path + "_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

def combine_datasets(datasets):
    if not datasets:
        return None
    
    combined_samples = torch.cat([d['samples'] for d in datasets], dim=0)
    combined_labels = torch.cat([d['labels'] for d in datasets], dim=0)
    
    return {
        "samples": combined_samples,
        "labels": combined_labels
    }


# ============= Main Data Generator Functions =============
def data_generator_pretraining_only(sourcedata_path, configs, training_mode, subset=True, 
                                   source_train_ratio=0.8, force_rebuild=False):
    """
    Data generator for pre-training only using all power levels (1.0kW, 1.5kW, 3.0kW)
    """
    print(f"\n{'='*50}")
    print(f"PRE-TRAINING DATA GENERATOR")
    print(f"SOURCE DATA PATH: {sourcedata_path}")
    print(f"SOURCE TRAIN RATIO: {source_train_ratio}")
    
    # Use all three power levels for pre-training
    source_power_levels = ['1.0', '1.5', '3.0']

    source_cache_base = os.path.join(sourcedata_path, "cache")
    os.makedirs(source_cache_base, exist_ok=True)
    source_power_str = "_".join(source_power_levels)
    source_cache_path = os.path.join(source_cache_base, f"pretraining_power_{source_power_str}_train_ratio_{source_train_ratio:.4f}")
    
    source_params = {'power_levels': source_power_levels, 'train_ratio': source_train_ratio, 'type': 'pretraining'}
    source_should_rebuild = force_rebuild or clear_cache_if_params_changed(source_cache_path, source_params)
    
    if source_should_rebuild or not os.path.exists(source_cache_path + ".pt"):
        print(f"Preparing pre-training data with power_levels={source_power_levels}, train_ratio={source_train_ratio}...")
        
        all_source_train_datasets = []
        for power_level in source_power_levels:
            base_tdms_path = getattr(configs, 'tdms_source_path', '/root/autodl-tmp')
            
            print(f"Loading source data for power level: {power_level}")
            train_dataset, _, _ = prepare_single_power_data(
                power_level, base_tdms_path, train_ratio=source_train_ratio
            )
            all_source_train_datasets.append(train_dataset)
        
        train_dataset = combine_datasets(all_source_train_datasets)
        
        save_with_metadata(source_cache_path, {'train': train_dataset}, source_params)
        print(f"Pre-training data saved to {source_cache_path}")
    else:
        print(f"Loading cached pre-training data from {source_cache_path} (matches params={source_params})")
        train_dataset = torch.load(source_cache_path + ".pt")['train']
    
    # Create dataset objects
    train_dataset = Load_Dataset(train_dataset, configs, training_mode, 
                                power_levels=source_power_levels,
                                target_dataset_size=configs.batch_size, subset=subset)
    
    # Create data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)
    
    print(f"Pre-training data loader created with {len(train_dataset)} samples")
    return train_loader


def data_generator_finetuning_only(configs, training_mode, subset=True, 
                                  danfoss_train_ratio=1.0, force_rebuild=False):
    """
    Data generator for fine-tuning only using Danfoss dataset
    All operating conditions are used for fine-tuning
    """
    print(f"\n{'='*50}")
    print(f"FINE-TUNING DATA GENERATOR (Danfoss)")
    print(f"DANFOSS TRAIN RATIO (file sampling): {danfoss_train_ratio}")
    print(f"Data Split: Fine-tune Train=8% (10% of 80%) | Test=20% | Discard=72%")

    cache_base = "/root/autodl-tmp/danfoss_cache"
    os.makedirs(cache_base, exist_ok=True)
    cache_path = os.path.join(cache_base, f"danfoss_finetuning_ratio_{danfoss_train_ratio:.4f}_split_8_20")

    danfoss_params = {
        'train_ratio': danfoss_train_ratio,
        'type': 'danfoss_finetuning',
        'split': '8_20'
    }
    should_rebuild = force_rebuild or clear_cache_if_params_changed(cache_path, danfoss_params)

    if should_rebuild or not os.path.exists(cache_path + ".pt"):
        print(f"Preparing Danfoss fine-tuning data with train_ratio={danfoss_train_ratio}...")
        
        # Load Danfoss data
        X_data, y_data = load_danfoss_data(train_ratio=danfoss_train_ratio)
        
        if X_data.size == 0 or y_data.size == 0:
            raise ValueError("No Danfoss data loaded. Check file paths and settings.")
        
        # Convert to tensors
        X_data = torch.tensor(X_data, dtype=torch.float32)
        y_data = torch.tensor(y_data, dtype=torch.long)
        
        total_size = len(X_data)
        print(f"Total Danfoss samples before splitting: {total_size}")

        test_size = int(total_size * 0.2)
        remaining_size = total_size - test_size
        finetune_train_size = int(remaining_size * 0.1)

        print(f"Splitting: Fine-tune Train={finetune_train_size} ({finetune_train_size/total_size*100:.1f}%) | "
              f"Test={test_size} ({test_size/total_size*100:.1f}%)")
      
        indices = np.random.permutation(total_size)

        test_indices = indices[-test_size:]

        remaining_indices = indices[:-test_size]
        finetune_train_indices = remaining_indices[:finetune_train_size]

        finetune_train_dataset = {
            "samples": X_data[finetune_train_indices],
            "labels": y_data[finetune_train_indices]
        }

        test_dataset = {
            "samples": X_data[test_indices],
            "labels": y_data[test_indices]
        }

        save_with_metadata(cache_path, {
            'finetune_train': finetune_train_dataset,
            'test': test_dataset,
            'total_size': total_size
        }, danfoss_params)
        print(f"Danfoss data saved to {cache_path}")
    else:
        print(f"Loading cached Danfoss data from {cache_path} (matches params={danfoss_params})")
        cache_data = torch.load(cache_path + ".pt")
        finetune_train_dataset = cache_data['finetune_train']
        test_dataset = cache_data['test']
        total_size = cache_data['total_size']

    # Create dataset objects - no power level mapping needed for Danfoss data
    finetune_train_dataset_obj = Load_Dataset(finetune_train_dataset, configs, training_mode, 
                                             power_levels=[],  # Empty for Danfoss data
                                             target_dataset_size=configs.target_batch_size, subset=subset)
    
    test_dataset_obj = Load_Dataset(test_dataset, configs, training_mode, 
                                   power_levels=[],  # Empty for Danfoss data
                                   target_dataset_size=configs.target_batch_size, subset=subset)

    # Create data loaders
    finetune_loader = torch.utils.data.DataLoader(
        dataset=finetune_train_dataset_obj, 
        batch_size=configs.target_batch_size,
        shuffle=True, 
        drop_last=configs.drop_last,
        num_workers=0
    )
    
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset_obj, 
        batch_size=configs.target_batch_size,
        shuffle=False, 
        drop_last=False,
        num_workers=0
    )

    print(f"\n✅ Danfoss Fine-tuning Data Loaders Created:")
    print(f"  - Fine-tune Train: {len(finetune_train_dataset_obj)} samples "
          f"({len(finetune_train_dataset_obj)/total_size*100:.1f}% of total)")
    print(f"  - Test: {len(test_dataset_obj)} samples "
          f"({len(test_dataset_obj)/total_size*100:.1f}% of total)")

    return finetune_loader, test_loader


def data_generator_combined(sourcedata_path, configs, training_mode, subset=True, 
                           source_train_ratio=0.8, danfoss_train_ratio=1.0, force_rebuild=False):
    """
    Combined data generator for both pre-training and fine-tuning
    Pre-training: 1.0kW, 1.5kW, 3.0kW TDMS data
    Fine-tuning: Danfoss data from all operating conditions
    """
    print(f"\n{'='*50}")
    print(f"COMBINED DATA GENERATOR")
    print("Pre-training: TDMS data (1.0kW, 1.5kW, 3.0kW)")
    print("Fine-tuning: Danfoss data (all operating conditions)")
    
    # Get pre-training data loader
    train_loader = data_generator_pretraining_only(
        sourcedata_path, configs, training_mode, subset, 
        source_train_ratio, force_rebuild
    )
    
    # Get fine-tuning data loaders
    finetune_loader, test_loader = data_generator_finetuning_only(
        configs, training_mode, subset, 
        danfoss_train_ratio, force_rebuild
    )
    
    print(f"\nCOMBINED DATA GENERATORS READY:")
    print(f"  - Pre-training loader: {len(train_loader.dataset)} samples")
    print(f"  - Fine-tuning loader: {len(finetune_loader.dataset)} samples")
    print(f"  - Test loader: {len(test_loader.dataset)} samples")
    
    return train_loader, finetune_loader, test_loader


# ============= Legacy Functions (for compatibility) =============
def data_generator_only_ft(targetdata_path, configs, training_mode, power_levels, subset=True, 
                          train_ratio=0.02, force_rebuild=False):
    """
    Legacy function for fine-tuning only with TDMS data (kept for compatibility)
    Use data_generator_finetuning_only for Danfoss data instead
    """
    print(f"WARNING: Using legacy TDMS fine-tuning function. Consider using data_generator_finetuning_only for Danfoss data.")
    
    print(f"\n{'='*50}")
    print(f"TARGET DATA PATH: {targetdata_path}")
    print(f"CONFIG TARGET POWER LEVELS: {power_levels}")
    print(f"TRAIN RATIO: {train_ratio}")
    
    cache_base = os.path.join(targetdata_path, "cache")
    os.makedirs(cache_base, exist_ok=True)
    
    power_str = "_".join(power_levels)
    cache_name = f"power_{power_str}_train_ratio_{train_ratio:.4f}"
    cache_path = os.path.join(cache_base, cache_name)
    
    params = {'power_levels': power_levels, 'train_ratio': train_ratio}
    should_rebuild = force_rebuild or clear_cache_if_params_changed(cache_path, params)
    
    all_train_datasets = []
    all_val_datasets = []
    all_test_datasets = []
    
    if should_rebuild or not os.path.exists(cache_path + ".pt"):
        print(f"Preparing target data splits with power_levels={power_levels}, train_ratio={train_ratio}...")
        
        for power_level in power_levels:
            base_tdms_path = getattr(configs, 'tdms_target_path', '/root/autodl-tmp')
            
            print(f"Loading data for power level: {power_level}")
            train_dataset, val_dataset, test_dataset = prepare_single_power_data(
                power_level, base_tdms_path, train_ratio=train_ratio
            )
            
            all_train_datasets.append(train_dataset)
            all_val_datasets.append(val_dataset)
            all_test_datasets.append(test_dataset)
        
        train_dataset = combine_datasets(all_train_datasets)
        val_dataset = combine_datasets(all_val_datasets)
        test_dataset = combine_datasets(all_test_datasets)
        
        save_with_metadata(cache_path, {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }, params)
        print(f"Target data saved to {cache_path}")
    else:
        print(f"Loading cached target data from {cache_path} (matches params={params})")
        cache_data = torch.load(cache_path + ".pt")
        train_dataset = cache_data['train']
        val_dataset = cache_data['val']
        test_dataset = cache_data['test']
    
    # Create dataset objects - 传递power_levels用于标签映射
    train_dataset = Load_Dataset(train_dataset, configs, training_mode, 
                                power_levels=power_levels,  # 传递多个功率级别
                                target_dataset_size=configs.batch_size, subset=subset)
    finetune_dataset = Load_Dataset(val_dataset, configs, training_mode, 
                                   power_levels=power_levels,
                                   target_dataset_size=configs.target_batch_size, subset=subset)
    
    if test_dataset['labels'].shape[0] > 10 * configs.target_batch_size:
        test_dataset = Load_Dataset(test_dataset, configs, training_mode, 
                                   power_levels=power_levels,
                                   target_dataset_size=configs.target_batch_size*10, subset=subset)
    else:
        test_dataset = Load_Dataset(test_dataset, configs, training_mode, 
                                   power_levels=power_levels,
                                   target_dataset_size=configs.target_batch_size, subset=subset)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)

    valid_loader = torch.utils.data.DataLoader(dataset=finetune_dataset, batch_size=configs.target_batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.target_batch_size,
                                              shuffle=True, drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader


def data_generator(sourcedata_path, targetdata_path, configs, training_mode, 
                  source_power_levels, target_power_levels, 
                  subset=True, source_train_ratio=0.8, target_train_ratio=0.02, force_rebuild=False):
    """
    Legacy function for both source and target TDMS datasets (kept for compatibility)
    Use data_generator_combined for TDMS pre-training + Danfoss fine-tuning instead
    """
    print(f"WARNING: Using legacy TDMS-only function. Consider using data_generator_combined for TDMS + Danfoss.")
    
    # Load source dataset (for pre-training)
    source_cache_base = os.path.join(sourcedata_path, "cache")
    os.makedirs(source_cache_base, exist_ok=True)
    source_power_str = "_".join(source_power_levels)
    source_cache_path = os.path.join(source_cache_base, f"source_power_{source_power_str}_train_ratio_{source_train_ratio:.4f}")
    
    source_params = {'power_levels': source_power_levels, 'train_ratio': source_train_ratio}
    source_should_rebuild = force_rebuild or clear_cache_if_params_changed(source_cache_path, source_params)
    
    all_source_train_datasets = []
    
    if source_should_rebuild or not os.path.exists(source_cache_path + ".pt"):
        print(f"Preparing source data splits with power_levels={source_power_levels}, train_ratio={source_train_ratio}...")
        for power_level in source_power_levels:
            base_tdms_path = getattr(configs, 'tdms_source_path', '/root/autodl-tmp')
            
            print(f"Loading source data for power level: {power_level}")
            train_dataset, _, _ = prepare_single_power_data(
                power_level, base_tdms_path, train_ratio=source_train_ratio
            )
            all_source_train_datasets.append(train_dataset)
        
        train_dataset = combine_datasets(all_source_train_datasets)
        
        save_with_metadata(source_cache_path, {'train': train_dataset}, source_params)
        print(f"Source data saved to {source_cache_path}")
    else:
        print(f"Loading cached source data from {source_cache_path} (matches params={source_params})")
        train_dataset = torch.load(source_cache_path + ".pt")['train']
    
    # Load target dataset (for fine-tuning)
    target_cache_base = os.path.join(targetdata_path, "cache")
    os.makedirs(target_cache_base, exist_ok=True)
    target_power_str = "_".join(target_power_levels)
    target_cache_path = os.path.join(target_cache_base, f"target_power_{target_power_str}_train_ratio_{target_train_ratio:.4f}")
    
    target_params = {'power_levels': target_power_levels, 'train_ratio': target_train_ratio}
    target_should_rebuild = force_rebuild or clear_cache_if_params_changed(target_cache_path, target_params)
    
    all_target_val_datasets = []
    all_target_test_datasets = []
    
    if target_should_rebuild or not os.path.exists(target_cache_path + ".pt"):
        print(f"Preparing target data splits with power_levels={target_power_levels}, train_ratio={target_train_ratio}...")
        for power_level in target_power_levels:
            base_tdms_path = getattr(configs, 'tdms_target_path', '/root/autodl-tmp')
            
            print(f"Loading target data for power level: {power_level}")
            _, val_dataset, test_dataset = prepare_single_power_data(
                power_level, base_tdms_path, train_ratio=target_train_ratio
            )
            all_target_val_datasets.append(val_dataset)
            all_target_test_datasets.append(test_dataset)
        
        val_dataset = combine_datasets(all_target_val_datasets)
        test_dataset = combine_datasets(all_target_test_datasets)
        
        save_with_metadata(target_cache_path, {
            'val': val_dataset,
            'test': test_dataset
        }, target_params)
        print(f"Target data saved to {target_cache_path}")
    else:
        print(f"Loading cached target data from {target_cache_path} (matches params={target_params})")
        cache_data = torch.load(target_cache_path + ".pt")
        val_dataset = cache_data['val']
        test_dataset = cache_data['test']

    # Create dataset objects
    train_dataset = Load_Dataset(train_dataset, configs, training_mode, 
                                power_levels=source_power_levels,
                                target_dataset_size=configs.batch_size, subset=subset)
    finetune_dataset = Load_Dataset(val_dataset, configs, training_mode, 
                                   power_levels=target_power_levels,
                                   target_dataset_size=configs.target_batch_size, subset=subset)
    
    if test_dataset['labels'].shape[0] > 10 * configs.target_batch_size:
        test_dataset = Load_Dataset(test_dataset, configs, training_mode, 
                                   power_levels=target_power_levels,
                                   target_dataset_size=configs.target_batch_size*10, subset=subset)
    else:
        test_dataset = Load_Dataset(test_dataset, configs, training_mode, 
                                   power_levels=target_power_levels,
                                   target_dataset_size=configs.target_batch_size, subset=subset)

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)

    valid_loader = torch.utils.data.DataLoader(dataset=finetune_dataset, batch_size=configs.target_batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.target_batch_size,
                                              shuffle=True, drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader
