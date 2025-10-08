import torch
from torch.utils.data import DataLoader, Dataset, random_split
import os
import numpy as np
from nptdms import TdmsFile
import json
from datetime import datetime

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode, power_levels, target_dataset_size=64, subset=False):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode
        self.power_levels = power_levels
        X_train = dataset["samples"]
        y_train = dataset["labels"]
        
        # Create power level mapping
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

        # Apply dynamic label mapping
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


# Helper functions for TDMS data processing
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

def segment(data_list, label_list):
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

def prepare_single_power_data(power_level, base_tdms_path='/root/autodl-tmp', train_ratio=0.02, val_ratio=0.1):
    """
    Prepare data from a single power level (for source or target dataset)
    """
    print(f"Processing power level: {power_level} from path: {base_tdms_path}")
    
    # Load and process data for the specified power level
    c_data, labels = load_power_data(power_level, 'current', base_tdms_path)
    c_data, labels = segment(c_data, labels)
    print(f"Current data shape: {c_data.shape}, Labels: {len(labels)}")
    
    v_data, _ = load_power_data(power_level, 'vibration', base_tdms_path)
    v_data, _ = segment(v_data, _)
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
    test_size = int(total_size * 0.2)
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


def data_generator_only_ft(targetdata_path, configs, training_mode, power_levels, subset=True, 
                          train_ratio=0.02, force_rebuild=False):
    """
    Data generator for fine-tuning only (matches official format)
    power_levels: list of power levels to use for fine-tuning (e.g., ['1.5', '3.0'])
    """
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
    
    # Create dataset objects
    train_dataset = Load_Dataset(train_dataset, configs, training_mode, 
                                power_levels=power_levels,
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
    Data generator for both source and target datasets (matches official format)
    source_power_levels: list of power levels for pre-training (e.g., ['1.0'])
    target_power_levels: list of power levels for fine-tuning (e.g., ['1.5', '3.0'])
    """
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


def combine_datasets(datasets):
    if not datasets:
        return None
    
    combined_samples = torch.cat([d['samples'] for d in datasets], dim=0)
    combined_labels = torch.cat([d['labels'] for d in datasets], dim=0)
    
    return {
        "samples": combined_samples,
        "labels": combined_labels
    }
