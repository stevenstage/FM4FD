# trainer.py (optimized)
import os
import sys
sys.path.append("..")
import json
import time
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.decomposition import PCA
from sklearn.utils.extmath import randomized_svd
from scipy.linalg import svd
from scipy.special import logsumexp

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, f1_score, recall_score

# local imports (adjust paths if needed)
from model import Model
from model import target_classifier
from logme import LogME

# -------------------------
# Utility: feature_reduce
# -------------------------
def feature_reduce(features: np.ndarray, f: int = None, whiten: bool = True, random_state: int = 1919):
    """
    PCA reduce features to f components.
    features: (N, D)
    If f is None -> return original features.
    Ensures f <= D.
    whiten=True recommended when using NCC with unit-cov assumption.
    """
    if f is None:
        return features
    if features.ndim != 2:
        raise ValueError("feature_reduce expects 2D array (N, D)")
    N, D = features.shape
    if f > D:
        f = D
    pca = PCA(n_components=f, svd_solver='randomized', random_state=random_state, whiten=whiten)
    return pca.fit_transform(features.astype(np.float32))


# -------------------------
# NCC streaming & helpers
# -------------------------
def _ncc_score_streaming(Z: np.ndarray, y_labels: np.ndarray):
    """
    Compute the NCC score in a streaming / memory-friendly way.
    Z: (N, p) numpy array (assumed float32)
    y_labels: (N,) int array
    returns: average posterior probability for true class (float)
    """
    N = Z.shape[0]
    labels = np.unique(y_labels)
    # streaming log-sum-exp accumulator per sample
    logsum = np.full(N, -np.inf, dtype=np.float32)
    logp_true = np.empty(N, dtype=np.float32)
    for c in labels:
        mask = (y_labels == c)
        Nc = int(mask.sum())
        if Nc == 0:
            continue
        mu = Z[mask].mean(axis=0)  # (p,)
        diff = Z - mu  # (N, p)
        sqdist = np.sum(diff * diff, axis=1, dtype=np.float32)  # (N,)
        logp_c = -0.5 * sqdist + np.log(float(Nc) / float(N))
        # streaming logsumexp update
        m = np.maximum(logsum, logp_c)
        # handle -inf
        valid = (m != -np.inf)
        if np.any(valid):
            # safe update for entries where at least one is finite
            logsum[valid] = m[valid] + np.log(np.exp(logsum[valid] - m[valid]) + np.exp(logp_c[valid] - m[valid]))
        else:
            # first class: logsum = logp_c
            logsum = logp_c
        # store true-class logp
        logp_true[mask] = logp_c[mask]
    # compute posterior probability for true labels
    probs_true = np.exp(logp_true - logsum)
    return float(np.mean(probs_true))


# -------------------------
# Optimized evaluate_ncc
# -------------------------
def evaluate_ncc_optimized(model, dataloader, device, logger, stage="", max_samples=2000, pca_dim=128, divide=8, seed=1919):
    """
    Optimized NCC evaluation that avoids OOM by:
      - reservoir sampling during feature extraction (fixed peak memory)
      - PCA reduction (randomized) with optional whiten
      - randomized_svd on reduced matrix -> build Z = U * s
      - segment Z and compute NCC in Z-space (no full reconstruction)
      - streaming log-sum-exp to avoid N x C allocation
    """
    logger.debug(f"Evaluating NCC score (optimized) at {stage}...")
    start_time = time.time()
    rng = np.random.RandomState(seed)

    # reservoir sampling while extracting features
    buffer_size = max_samples
    reservoir_feats = []
    reservoir_labels = []
    n_seen = 0

    model.eval()
    with torch.no_grad():
        for data, labels in dataloader:
            data = data.float().to(device)
            h, _ = model(data)
            feat = h.cpu().detach().numpy()
            # ensure 2D
            if feat.ndim > 2:
                feat = feat.reshape(feat.shape[0], -1)
            lbls = labels.cpu().numpy()
            for i in range(feat.shape[0]):
                n_seen += 1
                if len(reservoir_feats) < buffer_size:
                    reservoir_feats.append(feat[i].astype(np.float32))
                    reservoir_labels.append(int(lbls[i]))
                else:
                    j = rng.randint(0, n_seen)
                    if j < buffer_size:
                        reservoir_feats[j] = feat[i].astype(np.float32)
                        reservoir_labels[j] = int(lbls[i])

    if len(reservoir_feats) == 0:
        raise RuntimeError("No features collected for NCC evaluation")

    X = np.stack(reservoir_feats, axis=0).astype(np.float32)  # (N, D)
    y = np.array(reservoir_labels, dtype=int)
    N = X.shape[0]
    logger.debug(f"Collected {N} samples (max_samples={max_samples}) for NCC")

    # PCA reduce (randomized) - whiten to make unit-cov assumption reasonable
    Xr = X
    if pca_dim is not None and 0 < pca_dim < X.shape[1]:
        pca = PCA(n_components=pca_dim, svd_solver='randomized', random_state=seed, whiten=False)
        Xr = pca.fit_transform(X)
        del X
        gc.collect()

    # randomized SVD on Xr to get U, s, VT (k <= pca_dim)
    k = min(Xr.shape[0], Xr.shape[1])
    try:
        U, s, VT = randomized_svd(Xr, n_components=k, random_state=seed)
    except Exception as e:
        logger.debug(f"randomized_svd failed ({e}); fallback to scipy.linalg.svd")
        U, s, VT = svd(Xr, full_matrices=False)
    U = U.astype(np.float32)
    s = s.astype(np.float32)
    # spectral coordinates
    Z = U * s.reshape(1, -1)  # (N, k)
    del U, VT
    gc.collect()

    # Segment Z and compute per-segment NCC score
    f = Z.shape[1]
    divide = max(1, divide)
    partition_size = max(1, f // divide)
    sum_s = float(np.sum(s)) if s.size > 0 else 0.0
    nccscore_list = {}
    ratio_list = {}
    for i in range(divide):
        start = i * partition_size
        end = start + partition_size if i < divide - 1 else f
        if start >= end:
            nccscore_list[i] = 0.0
            ratio_list[i] = 0.0
            continue
        Z_seg = Z[:, start:end]  # (N, psize)
        nccscore_list[i] = _ncc_score_streaming(Z_seg, y)
        ratio_list[i] = float(np.sum(s[start:end]) / sum_s) if sum_s > 0 else 0.0

    weighted_ncc = sum(nccscore_list[i] * ratio_list[i] for i in range(divide))

    # cleanup
    del Z, s
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

    ncc_result = {"time": time.time() - start_time, "ncc": nccscore_list, "ratio": ratio_list}
    logger.debug(f"NCC segment scores at {stage}: {nccscore_list}")
    logger.debug(f"Singular value ratios at {stage}: {ratio_list}")
    logger.debug(f"Weighted average NCC score at {stage}: {weighted_ncc:.6f}")
    logger.debug(f"NCC evaluation time: {ncc_result['time']:.4f} seconds")
    return ncc_result, weighted_ncc

def build_model(args, lr, configs, device, chkpoint=None):
    model = Model(configs, args).to(device)
    if chkpoint:
        pretrained_dict = chkpoint.get("model_state_dict", {})
        model_dict = model.state_dict()
        # filter keys that exist in model_dict and shape-match
        filtered = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
    classifier = target_classifier(configs).to(device)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(configs.beta1, configs.beta2), weight_decay=0)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, betas=(configs.beta1, configs.beta2),
                                            weight_decay=0)
    model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optimizer, T_max=args.finetune_epoch)
    return model, classifier, model_optimizer, classifier_optimizer, model_scheduler


# Add these imports to your trainer.py
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def visualize_frenorm_knowledge_base(frenorm_layer, save_path, stage="", figsize=(15, 10)):
    """
    Visualize FreNormLayer_KB knowledge base
    """
    if frenorm_layer is None:
        return None
    
    kb = frenorm_layer.kb.data.cpu().numpy()  # [n_knlg, embed_dim]
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.suptitle(f'FreNormLayer_KB Knowledge Base - {stage}', fontsize=16)
    
    # 1. Knowledge base magnitude heatmap
    magnitude = np.abs(kb)
    im1 = axes[0,0].imshow(magnitude, aspect='auto', cmap='viridis')
    axes[0,0].set_title('KB Magnitude')
    axes[0,0].set_xlabel('Frequency Bins')
    axes[0,0].set_ylabel('Knowledge Entries')
    plt.colorbar(im1, ax=axes[0,0])
    
    # 2. Knowledge base real part
    im2 = axes[0,1].imshow(np.real(kb), aspect='auto', cmap='RdBu_r')
    axes[0,1].set_title('KB Real Part')
    axes[0,1].set_xlabel('Frequency Bins')
    axes[0,1].set_ylabel('Knowledge Entries')
    plt.colorbar(im2, ax=axes[0,1])
    
    # 3. Knowledge base imaginary part
    im3 = axes[0,2].imshow(np.imag(kb), aspect='auto', cmap='RdBu_r')
    axes[0,2].set_title('KB Imaginary Part')
    axes[0,2].set_xlabel('Frequency Bins')
    axes[0,2].set_ylabel('Knowledge Entries')
    plt.colorbar(im3, ax=axes[0,2])
    
    # 4. Average frequency response
    avg_magnitude = np.mean(magnitude, axis=0)
    std_magnitude = np.std(magnitude, axis=0)
    freqs = np.arange(len(avg_magnitude))
    axes[1,0].plot(freqs, avg_magnitude, 'b-', linewidth=2)
    axes[1,0].fill_between(freqs, avg_magnitude - std_magnitude, 
                          avg_magnitude + std_magnitude, alpha=0.3)
    axes[1,0].set_title('Average Magnitude Response')
    axes[1,0].set_xlabel('Frequency Bin')
    axes[1,0].set_ylabel('Magnitude')
    axes[1,0].grid(True)
    
    # 5. Individual filter responses (first 5)
    for i in range(min(5, kb.shape[0])):
        axes[1,1].plot(freqs, magnitude[i], label=f'KB-{i}', alpha=0.7)
    axes[1,1].set_title('Individual KB Entries')
    axes[1,1].set_xlabel('Frequency Bin')
    axes[1,1].set_ylabel('Magnitude')
    axes[1,1].legend()
    axes[1,1].grid(True)
    
    # 6. Knowledge base similarity matrix
    kb_flat = kb.reshape(kb.shape[0], -1)
    similarity_matrix = np.corrcoef(np.abs(kb_flat))
    im6 = axes[1,2].imshow(similarity_matrix, cmap='coolwarm', aspect='equal')
    axes[1,2].set_title('KB Entry Similarity')
    axes[1,2].set_xlabel('KB Entry Index')
    axes[1,2].set_ylabel('KB Entry Index')
    plt.colorbar(im6, ax=axes[1,2])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()  # Close to save memory
    
    return {
        'magnitude': magnitude,
        'avg_response': avg_magnitude,
        'similarity': similarity_matrix
    }

def analyze_frenorm_with_sample(model, sample_data, save_path, stage=""):
    """
    Analyze FreNormLayer_KB behavior with sample data
    """
    if not hasattr(model, 'fre_norm_encoder') or model.fre_norm_encoder is None:
        return None
    
    frenorm_layer = model.fre_norm_encoder
    device = next(model.parameters()).device
    
    # Ensure sample_data is on correct device
    if isinstance(sample_data, torch.Tensor):
        sample_data = sample_data.to(device)
    else:
        sample_data = torch.tensor(sample_data, dtype=torch.float32).to(device)
    
    if sample_data.dim() == 3:
        sample_data = sample_data.reshape(-1, sample_data.shape[-1])
    
    model.eval()
    with torch.no_grad():
        # Get FreNormLayer output and weights
        output, w = frenorm_layer(sample_data)
        
        # Get input FFT for comparison
        input_fft = torch.fft.rfft(sample_data, dim=-1, norm='ortho')
        output_fft = torch.fft.rfft(output, dim=-1, norm='ortho')
        
        # Convert to numpy
        input_fft_np = input_fft.cpu().numpy()
        output_fft_np = output_fft.cpu().numpy()
        w_np = w.cpu().numpy()
        
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'FreNormLayer_KB Processing Analysis - {stage}', fontsize=16)
    
    # Use first sample for visualization
    sample_idx = 0
    freqs = np.arange(input_fft_np.shape[-1])
    
    # Input vs Output magnitude spectrum
    axes[0,0].plot(freqs, np.abs(input_fft_np[sample_idx]), 'b-', label='Input', linewidth=2)
    axes[0,0].plot(freqs, np.abs(output_fft_np[sample_idx]), 'r--', label='Output', linewidth=2)
    axes[0,0].set_title('Input vs Output Spectrum')
    axes[0,0].set_xlabel('Frequency Bin')
    axes[0,0].set_ylabel('Magnitude')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # Learned weights magnitude
    axes[0,1].plot(freqs, np.abs(w_np[sample_idx]), 'g-', linewidth=2)
    axes[0,1].set_title('Learned Weights |w|')
    axes[0,1].set_xlabel('Frequency Bin')
    axes[0,1].set_ylabel('Magnitude')
    axes[0,1].grid(True)
    
    # Transfer function effect (output/input ratio)
    transfer_ratio = np.abs(output_fft_np[sample_idx]) / (np.abs(input_fft_np[sample_idx]) + 1e-8)
    axes[0,2].plot(freqs, transfer_ratio, 'm-', linewidth=2)
    axes[0,2].set_title('Transfer Function (|Output|/|Input|)')
    axes[0,2].set_xlabel('Frequency Bin')
    axes[0,2].set_ylabel('Ratio')
    axes[0,2].grid(True)
    
    # Phase analysis
    axes[1,0].plot(freqs, np.angle(input_fft_np[sample_idx]), 'b-', label='Input Phase')
    axes[1,0].plot(freqs, np.angle(output_fft_np[sample_idx]), 'r--', label='Output Phase')
    axes[1,0].set_title('Phase Analysis')
    axes[1,0].set_xlabel('Frequency Bin')
    axes[1,0].set_ylabel('Phase (rad)')
    axes[1,0].legend()
    axes[1,0].grid(True)
    
    # Weights heatmap (multiple samples)
    n_samples_viz = min(10, w_np.shape[0])
    im = axes[1,1].imshow(np.abs(w_np[:n_samples_viz]), aspect='auto', cmap='viridis')
    axes[1,1].set_title('Weights Heatmap (Multiple Samples)')
    axes[1,1].set_xlabel('Frequency Bin')
    axes[1,1].set_ylabel('Sample Index')
    plt.colorbar(im, ax=axes[1,1])
    
    # Statistical analysis of weights
    w_stats = {
        'mean': np.mean(np.abs(w_np), axis=0),
        'std': np.std(np.abs(w_np), axis=0),
        'max': np.max(np.abs(w_np), axis=0),
        'min': np.min(np.abs(w_np), axis=0)
    }
    
    axes[1,2].plot(freqs, w_stats['mean'], 'g-', label='Mean', linewidth=2)
    axes[1,2].fill_between(freqs, w_stats['mean'] - w_stats['std'], 
                          w_stats['mean'] + w_stats['std'], alpha=0.3, label='±1 STD')
    axes[1,2].plot(freqs, w_stats['max'], 'r:', label='Max', alpha=0.7)
    axes[1,2].plot(freqs, w_stats['min'], 'b:', label='Min', alpha=0.7)
    axes[1,2].set_title('Weight Statistics')
    axes[1,2].set_xlabel('Frequency Bin')
    axes[1,2].set_ylabel('Weight Magnitude')
    axes[1,2].legend()
    axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'input_spectrum': np.abs(input_fft_np[sample_idx]),
        'output_spectrum': np.abs(output_fft_np[sample_idx]),
        'weights': np.abs(w_np[sample_idx]),
        'weight_stats': w_stats,
        'transfer_ratio': transfer_ratio
    }

def compare_frenorm_states(pre_data, post_data, save_path):
    """
    Compare FreNormLayer_KB before and after fine-tuning
    """
    if pre_data is None or post_data is None:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('FreNormLayer_KB: Pre-training vs Fine-tuning Comparison', fontsize=16)
    
    freqs = np.arange(len(pre_data['avg_response']))
    
    # 1. Average response comparison
    axes[0,0].plot(freqs, pre_data['avg_response'], 'b-', label='Pre-training', linewidth=2)
    axes[0,0].plot(freqs, post_data['avg_response'], 'r-', label='Fine-tuning', linewidth=2)
    axes[0,0].set_title('Average KB Response Comparison')
    axes[0,0].set_xlabel('Frequency Bin')
    axes[0,0].set_ylabel('Average Magnitude')
    axes[0,0].legend()
    axes[0,0].grid(True)
    
    # 2. Response difference
    diff = post_data['avg_response'] - pre_data['avg_response']
    axes[0,1].plot(freqs, diff, 'g-', linewidth=2)
    axes[0,1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
    axes[0,1].set_title('Response Difference (Post - Pre)')
    axes[0,1].set_xlabel('Frequency Bin')
    axes[0,1].set_ylabel('Magnitude Difference')
    axes[0,1].grid(True)
    
    # 3. Magnitude comparison heatmap
    magnitude_diff = post_data['magnitude'] - pre_data['magnitude']
    im3 = axes[0,2].imshow(magnitude_diff, aspect='auto', cmap='RdBu_r')
    axes[0,2].set_title('KB Magnitude Changes')
    axes[0,2].set_xlabel('Frequency Bins')
    axes[0,2].set_ylabel('Knowledge Entries')
    plt.colorbar(im3, ax=axes[0,2])
    
    # 4. Similarity matrix changes
    sim_diff = post_data['similarity'] - pre_data['similarity']
    im4 = axes[1,0].imshow(sim_diff, cmap='RdBu_r', aspect='equal')
    axes[1,0].set_title('KB Similarity Changes')
    axes[1,0].set_xlabel('KB Entry Index')
    axes[1,0].set_ylabel('KB Entry Index')
    plt.colorbar(im4, ax=axes[1,0])
    
    # 5. Statistical summary
    stats_pre = [np.mean(pre_data['magnitude']), np.std(pre_data['magnitude']), 
                 np.max(pre_data['magnitude']), np.min(pre_data['magnitude'])]
    stats_post = [np.mean(post_data['magnitude']), np.std(post_data['magnitude']), 
                  np.max(post_data['magnitude']), np.min(post_data['magnitude'])]
    
    x_labels = ['Mean', 'Std', 'Max', 'Min']
    x_pos = np.arange(len(x_labels))
    
    axes[1,1].bar(x_pos - 0.2, stats_pre, 0.4, label='Pre-training', alpha=0.7)
    axes[1,1].bar(x_pos + 0.2, stats_post, 0.4, label='Fine-tuning', alpha=0.7)
    axes[1,1].set_title('Statistical Summary')
    axes[1,1].set_xlabel('Statistic')
    axes[1,1].set_ylabel('Value')
    axes[1,1].set_xticks(x_pos)
    axes[1,1].set_xticklabels(x_labels)
    axes[1,1].legend()
    axes[1,1].grid(True, axis='y')
    
    # 6. Cumulative change analysis
    cumulative_change = np.cumsum(np.abs(diff))
    axes[1,2].plot(freqs, cumulative_change, 'purple', linewidth=2)
    axes[1,2].set_title('Cumulative Absolute Change')
    axes[1,2].set_xlabel('Frequency Bin')
    axes[1,2].set_ylabel('Cumulative |Change|')
    axes[1,2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Modified Trainer function with integrated visualization
def Trainer(model,
            model_optimizer,
            model_scheduler,
            train_dl,
            valid_dl,
            test_dl,
            device,
            logger,
            args,
            configs,
            experiment_log_dir,
            seed):
    
    logger.debug(f"Training Mode: {args.training_mode}")
    os.makedirs(os.path.join(experiment_log_dir, f"saved_models"), exist_ok=True)
    os.makedirs(os.path.join(experiment_log_dir, f"frenorm_visualizations"), exist_ok=True)
    
    start_time = time.time()
    ncc_results = {}
    
    # Get sample data for FreNormLayer analysis
    sample_batch = next(iter(train_dl))
    sample_data = sample_batch[0][:4]  # Use first 4 samples
    
    # Initial FreNormLayer visualization (before any training)
    logger.debug("Visualizing initial FreNormLayer_KB state...")
    initial_kb_path = os.path.join(experiment_log_dir, "frenorm_visualizations", "kb_initial.png")
    initial_analysis_path = os.path.join(experiment_log_dir, "frenorm_visualizations", "analysis_initial.png")
    
    initial_kb_data = visualize_frenorm_knowledge_base(
        model.fre_norm_encoder, initial_kb_path, "Initial State"
    )
    initial_analysis_data = analyze_frenorm_with_sample(
        model, sample_data, initial_analysis_path, "Initial State"
    )
    
    if args.training_mode in ['pre_train', 'combined']:
        logger.debug("Pre-training started ....")
        for epoch in range(1, args.pretrain_epoch + 1):
            total_loss, total_cl_loss, total_rb_loss = model_pretrain(
                model, model_optimizer, model_scheduler, train_dl, configs, args, device
            )
            logger.debug(
                f'Pre-training Epoch: {epoch}\t Train Loss: {total_loss:.4f}\t CL Loss: {total_cl_loss:.4f}\t RB Loss: {total_rb_loss:.4f}\n')
            chkpoint = {
                'seed': seed,
                'epoch': epoch,
                'train_loss': total_loss,
                'model_state_dict': model.state_dict()
            }
            torch.save(chkpoint, os.path.join(experiment_log_dir, f"saved_models/", f'ckp_ep{epoch}.pt'))

        # Visualize after pre-training
        logger.debug("Visualizing FreNormLayer_KB after pre-training...")
        pretrain_kb_path = os.path.join(experiment_log_dir, "frenorm_visualizations", "kb_after_pretrain.png")
        pretrain_analysis_path = os.path.join(experiment_log_dir, "frenorm_visualizations", "analysis_after_pretrain.png")
        
        pretrain_kb_data = visualize_frenorm_knowledge_base(
            model.fre_norm_encoder, pretrain_kb_path, "After Pre-training"
        )
        pretrain_analysis_data = analyze_frenorm_with_sample(
            model, sample_data, pretrain_analysis_path, "After Pre-training"
        )
        
        # Create comparison visualization
        if initial_kb_data and pretrain_kb_data:
            pretrain_comparison_path = os.path.join(experiment_log_dir, "frenorm_visualizations", "comparison_pretrain.png")
            compare_frenorm_states(initial_kb_data, pretrain_kb_data, pretrain_comparison_path)

        # Evaluate NCC after pretrain
        if args.training_mode == 'pre_train':
            logger.debug("Evaluating pre-trained model with NCC score...")
            pretrain_ncc, pretrain_weighted_ncc = evaluate_ncc_optimized(model, valid_dl, device, logger, "pre-training",
                                                                         max_samples=2000, pca_dim=128, divide=8, seed=seed)
            ncc_results['pre_train'] = (pretrain_ncc, pretrain_weighted_ncc)

        if args.training_mode == 'combined':
            logger.debug("Preparing for fine-tuning phase...")
            chkpoint = torch.load(os.path.join(experiment_log_dir, "saved_models", f"ckp_ep{args.pretrain_epoch}.pt"))
            ft_model, ft_classifier, ft_model_optimizer, ft_classifier_optimizer, ft_scheduler = build_model(
                args, args.lr, configs, device, chkpoint
            )
        else:
            logger.debug("Pre-training completed. No fine-tuning.")
            end_time = time.time()
            training_time = end_time - start_time
            logger.debug(f"Total Training Time: {training_time:.2f} seconds")
            if ncc_results:
                _save_ncc_results(ncc_results, experiment_log_dir, logger)
            return None
    else:
        logger.debug(f"Skipping pre-training (training_mode={args.training_mode}) ...")
        ft_model = model
        ft_classifier = target_classifier(configs).to(device)
        ft_model_optimizer = torch.optim.Adam(ft_model.parameters(), lr=args.lr, betas=(configs.beta1, configs.beta2), weight_decay=0)
        ft_classifier_optimizer = torch.optim.Adam(ft_classifier.parameters(), lr=args.lr, betas=(configs.beta1, configs.beta2), weight_decay=0)
        ft_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=ft_model_optimizer, T_max=args.finetune_epoch)

    if args.training_mode in ['fine_tune', 'combined']:
        logger.debug("Fine-tuning started ....")
        best_performance = None
        best_logme_score = -float('inf')
        best_features = None
        best_labels = None

        ncc_results_during_finetune = {}
        
        # Store pre-fine-tuning state for comparison
        pre_finetune_kb_data = visualize_frenorm_knowledge_base(
            ft_model.fre_norm_encoder, 
            os.path.join(experiment_log_dir, "frenorm_visualizations", "kb_before_finetune.png"),
            "Before Fine-tuning"
        )

        for ep in range(1, args.finetune_epoch + 1):
            finetune_loader = train_dl if args.training_mode == 'fine_tune' else valid_dl
            if finetune_loader is None:
                logger.error("No fine-tuning data available!")
                break
            valid_loss, valid_acc, valid_auc, valid_prc, all_features, all_labels, F1 = model_finetune(
                ft_model, finetune_loader, device, ft_model_optimizer, ft_scheduler,
                classifier=ft_classifier, classifier_optimizer=ft_classifier_optimizer
            )

            # Periodic FreNormLayer visualization during fine-tuning
            if ep % 50 == 0 or ep == args.finetune_epoch:
                logger.debug(f"Visualizing FreNormLayer_KB at epoch {ep}...")
                epoch_kb_path = os.path.join(experiment_log_dir, "frenorm_visualizations", f"kb_epoch_{ep}.png")
                epoch_analysis_path = os.path.join(experiment_log_dir, "frenorm_visualizations", f"analysis_epoch_{ep}.png")
                
                epoch_kb_data = visualize_frenorm_knowledge_base(
                    ft_model.fre_norm_encoder, epoch_kb_path, f"Fine-tuning Epoch {ep}"
                )
                epoch_analysis_data = analyze_frenorm_with_sample(
                    ft_model, sample_data, epoch_analysis_path, f"Fine-tuning Epoch {ep}"
                )

            # NCC evaluation periodically
            if ep % 50 == 0 or ep == args.finetune_epoch:
                logger.debug(f"[NCC Evaluation] Evaluating model at epoch {ep}...")
                try:
                    ncc_result_ep, weighted_ncc_ep = evaluate_ncc_optimized(ft_model, test_dl, device, logger, f"epoch_{ep}",
                                                                            max_samples=2000, pca_dim=128, divide=8, seed=seed)
                    ncc_results_during_finetune[ep] = (ncc_result_ep, weighted_ncc_ep)
                    logger.debug(f"[NCC Evaluation] Epoch {ep} Weighted NCC Score: {weighted_ncc_ep:.4f}")
                except Exception as e:
                    logger.error(f"[NCC Evaluation] Failed at epoch {ep}: {str(e)}")

            if ep % args.log_epoch == 0:
                logger.debug(
                    f'\nEpoch : {ep}\t | \t  finetune Loss: {valid_loss:.4f}\t | \tAcc: {valid_acc:2.4f}\t | \tF1: {F1:0.4f}')
                test_loss, test_acc, test_auc, test_prc, emb_test, label_test, performance = model_test(
                    ft_model, test_dl, device, classifier=ft_classifier
                )
                if best_performance is None:
                    best_performance = performance
                else:
                    if performance[0] > best_performance[0]:
                        best_performance = performance
                        logger.debug(
                            'EP%s - Better Testing: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f' % (
                            ep, performance[0], performance[1], performance[2], performance[3]))
                        chkpoint = {
                            'seed': seed,
                            'epoch': ep,
                            'train_loss': valid_loss,
                            'model_state_dict': ft_model.state_dict(),
                            'classifier_state_dict': ft_classifier.state_dict()
                        }
                        torch.save(chkpoint, os.path.join(experiment_log_dir, f"saved_models/", f'ckp_best_ft.pt'))

        # Final FreNormLayer visualization after fine-tuning
        logger.debug("Creating final FreNormLayer_KB visualizations...")
        final_kb_path = os.path.join(experiment_log_dir, "frenorm_visualizations", "kb_final.png")
        final_analysis_path = os.path.join(experiment_log_dir, "frenorm_visualizations", "analysis_final.png")
        
        final_kb_data = visualize_frenorm_knowledge_base(
            ft_model.fre_norm_encoder, final_kb_path, "Final State"
        )
        final_analysis_data = analyze_frenorm_with_sample(
            ft_model, sample_data, final_analysis_path, "Final State"
        )
        
        # Create final comparison visualization
        if pre_finetune_kb_data and final_kb_data:
            final_comparison_path = os.path.join(experiment_log_dir, "frenorm_visualizations", "comparison_final.png")
            compare_frenorm_states(pre_finetune_kb_data, final_kb_data, final_comparison_path)
            logger.debug(f"FreNormLayer_KB comparison saved to {final_comparison_path}")

        # final NCC
        logger.debug("Evaluating final model with NCC score...")
        final_ncc, final_weighted_ncc = evaluate_ncc_optimized(ft_model, test_dl, device, logger, "final fine-tuning",
                                                               max_samples=2000, pca_dim=128, divide=8, seed=seed)
        ncc_results['fine_tune'] = (final_ncc, final_weighted_ncc)

        logger.debug("Fine-tuning ended ....")
        logger.debug("=" * 100)
        if best_performance is not None:
            logger.debug('Best Testing: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f' % (
                best_performance[0], best_performance[1], best_performance[2], best_performance[3]))
        else:
            logger.debug("No fine-tuning performance recorded.")
        logger.debug("=" * 100)

    end_time = time.time()
    training_time = end_time - start_time
    logger.debug(f"Total Training Time: {training_time:.2f} seconds")
    logger.debug(f"FreNormLayer_KB visualizations saved to: {os.path.join(experiment_log_dir, 'frenorm_visualizations')}")

    if ncc_results:
        _save_ncc_results(ncc_results, experiment_log_dir, logger)

    return best_performance if 'best_performance' in locals() else None


# -------------------------
# Save NCC results helper
# -------------------------
def _save_ncc_results(ncc_results, experiment_log_dir, logger):
    ncc_result_path = os.path.join(experiment_log_dir, "ncc_results.txt")
    try:
        with open(ncc_result_path, "w") as f:
            if 'pre_train' in ncc_results:
                pretrain_ncc, pretrain_weighted_ncc = ncc_results['pre_train']
                f.write(f"Pre-training weighted NCC score: {pretrain_weighted_ncc:.4f}\n")
                f.write("Pre-training NCC segment scores:\n")
                for i, score in pretrain_ncc["ncc"].items():
                    f.write(f"Segment {i}: {score:.4f} (ratio: {pretrain_ncc['ratio'][i]:.4f})\n")
                f.write("\n")

            if 'fine_tune' in ncc_results:
                final_ncc, final_weighted_ncc = ncc_results['fine_tune']
                f.write(f"Final fine-tuning weighted NCC score: {final_weighted_ncc:.4f}\n")
                f.write("Final fine-tuning NCC segment scores:\n")
                for i, score in final_ncc["ncc"].items():
                    f.write(f"Segment {i}: {score:.4f} (ratio: {final_ncc['ratio'][i]:.4f})\n")
                f.write("\n")

            if 'comparison' in ncc_results:
                comp = ncc_results['comparison']
                f.write(f"NCC score improvement: {comp['improvement']:.4f}\n")
        logger.debug(f"NCC results saved to {ncc_result_path}")
    except Exception as e:
        logger.error(f"Failed to save NCC results to {ncc_result_path}: {e}")


# -------------------------
# model_pretrain, model_finetune, model_test, linear_probe_eval
# -------------------------
def model_pretrain(model, model_optimizer, model_scheduler, train_loader, configs, args, device):
    total_loss = []
    total_cl_loss = []
    total_rb_loss = []
    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        model_optimizer.zero_grad()
        data, labels = data.float().to(device), labels.float().to(device)
        loss, loss_cl, loss_rb = model(data, pretrain=True)
        loss.backward()
        model_optimizer.step()
        total_loss.append(loss.item())
        total_cl_loss.append(loss_cl.item())
        total_rb_loss.append(loss_rb.item())
    total_loss = float(torch.tensor(total_loss).mean()) if total_loss else 0.0
    total_cl_loss = float(torch.tensor(total_cl_loss).mean()) if total_cl_loss else 0.0
    total_rb_loss = float(torch.tensor(total_rb_loss).mean()) if total_rb_loss else 0.0
    model_scheduler.step()
    return total_loss, total_cl_loss, total_rb_loss


def model_finetune(model, val_dl, device, model_optimizer, model_scheduler, classifier=None, classifier_optimizer=None):
    if classifier is None or classifier_optimizer is None:
        raise ValueError("classifier and classifier_optimizer must be provided to model_finetune")

    model.train()
    classifier.train()
    total_loss = []
    total_acc = []
    total_auc = []
    total_prc = []
    outs = np.array([], dtype=int)
    trgs = np.array([], dtype=int)
    all_features = []
    all_labels = []
    criterion = nn.CrossEntropyLoss()

    for data, labels in val_dl:
        model_optimizer.zero_grad()
        classifier_optimizer.zero_grad()
        data, labels = data.float().to(device), labels.long().to(device)
        h, z = model(data)
        fea_concat = h
        # auto flatten if needed
        feat = fea_concat
        if feat.ndim > 2:
            feat = feat.reshape(feat.shape[0], -1)
        all_features.append(feat.cpu().detach())
        all_labels.append(labels.cpu().detach())

        predictions = classifier(fea_concat)
        loss = criterion(predictions, labels)
        loss.backward()
        model_optimizer.step()
        classifier_optimizer.step()

        acc_bs = labels.eq(predictions.detach().argmax(dim=1)).float().mean()
        total_acc.append(acc_bs.cpu().item())

        pred_numpy = predictions.detach().cpu().numpy()
        try:
            auc_bs = roc_auc_score(F.one_hot(labels).detach().cpu().numpy(), pred_numpy, average="macro", multi_class="ovr")
        except Exception:
            auc_bs = 0.0
        try:
            prc_bs = average_precision_score(F.one_hot(labels).detach().cpu().numpy(), pred_numpy)
        except Exception:
            prc_bs = 0.0
        if auc_bs != 0:
            total_auc.append(auc_bs)
        if prc_bs != 0:
            total_prc.append(prc_bs)

        total_loss.append(loss.item())

        pred = predictions.max(1, keepdim=True)[1]
        outs = np.append(outs, pred.cpu().numpy().astype(int))
        trgs = np.append(trgs, labels.data.cpu().numpy().astype(int))

    if len(all_features) > 0:
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
    else:
        all_features = torch.tensor([])
        all_labels = torch.tensor([])

    if outs.size == 0:
        F1 = 0.0
    else:
        F1 = f1_score(trgs, outs, average='macro')

    total_loss = float(torch.tensor(total_loss).mean()) if total_loss else 0.0
    total_acc = float(np.mean(total_acc)) if total_acc else 0.0
    total_auc = float(np.mean(total_auc)) if total_auc else 0.0
    total_prc = float(np.mean(total_prc)) if total_prc else 0.0

    model_scheduler.step(total_loss)
    return total_loss, total_acc, total_auc, total_prc, all_features, all_labels, F1


def model_test(model, test_dl, device, classifier=None):
    model.eval()
    if classifier is not None:
        classifier.eval()
    total_loss = []
    total_acc = []
    total_auc = []
    total_prc = []
    total_precision, total_recall, total_f1 = [], [], []
    criterion = nn.CrossEntropyLoss()
    outs = np.array([], dtype=int)
    trgs = np.array([], dtype=int)
    emb_test_all = []
    with torch.no_grad():
        for data, labels in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)
            h, z = model(data)
            fea_concat = h
            fea_concat_flat = fea_concat.reshape(fea_concat.shape[0], -1)
            if classifier is not None:
                predictions_test = classifier(fea_concat)
            else:
                # if no classifier, use linear probe or skip
                raise ValueError("classifier must be provided for model_test")
            emb_test_all.append(fea_concat_flat)
            loss = criterion(predictions_test, labels)
            acc_bs = labels.eq(predictions_test.detach().argmax(dim=1)).float().mean()
            total_acc.append(acc_bs.cpu().item())

            pred_numpy = predictions_test.detach().cpu().numpy()
            labels_numpy = labels.detach().cpu().numpy()
            try:
                auc_bs = roc_auc_score(F.one_hot(labels).detach().cpu().numpy(), pred_numpy, average="macro", multi_class="ovr")
            except Exception:
                auc_bs = 0.0
            try:
                prc_bs = average_precision_score(F.one_hot(labels).detach().cpu().numpy(), pred_numpy, average="macro")
            except Exception:
                prc_bs = 0.0
            if auc_bs != 0:
                total_auc.append(auc_bs)
            if prc_bs != 0:
                total_prc.append(prc_bs)

            pred_numpy = np.argmax(pred_numpy, axis=1)
            precision = precision_score(labels_numpy, pred_numpy, average='macro')
            recall = recall_score(labels_numpy, pred_numpy, average='macro')
            F1 = f1_score(labels_numpy, pred_numpy, average='macro')
            total_precision.append(precision)
            total_recall.append(recall)
            total_f1.append(F1)

            total_loss.append(loss.item())
            pred = predictions_test.max(1, keepdim=True)[1]
            outs = np.append(outs, pred.cpu().numpy().astype(int))
            trgs = np.append(trgs, labels.data.cpu().numpy().astype(int))

    if outs.size == 0:
        raise RuntimeError("No predictions collected in model_test")
    labels_numpy_all = trgs
    pred_numpy_all = outs
    precision = precision_score(labels_numpy_all, pred_numpy_all, average='macro')
    recall = recall_score(labels_numpy_all, pred_numpy_all, average='macro')
    F1 = f1_score(labels_numpy_all, pred_numpy_all, average='macro')
    acc = accuracy_score(labels_numpy_all, pred_numpy_all)
    total_loss = float(torch.tensor(total_loss).mean()) if total_loss else 0.0
    total_acc = float(np.mean(total_acc)) if total_acc else 0.0
    total_auc = float(np.mean(total_auc)) if total_auc else 0.0
    total_prc = float(np.mean(total_prc)) if total_prc else 0.0
    performance = [acc * 100, precision * 100, recall * 100, F1 * 100, total_auc * 100, total_prc * 100]
    emb_test_all = torch.cat(tuple(emb_test_all))
    return total_loss, total_acc, total_auc, total_prc, emb_test_all, trgs, performance


def linear_probe_eval(model, train_dl, test_dl, device, configs):
    model.eval()
    classifier = target_classifier(configs).to(device)
    classifier_optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=configs.lr,
        betas=(configs.beta1, configs.beta2),
        weight_decay=0
    )
    classifier.train()
    for epoch in range(1, 11):
        for data, labels in train_dl:
            data, labels = data.float().to(device), labels.long().to(device)
            classifier_optimizer.zero_grad()
            with torch.no_grad():
                h, _ = model(data)
                fea_concat = h
            predictions = classifier(fea_concat)
            loss = nn.CrossEntropyLoss()(predictions, labels)
            loss.backward()
            classifier_optimizer.step()
    classifier.eval()
    total_loss = []
    total_acc = []
    total_auc = []
    total_prc = []
    total_precision, total_recall, total_f1 = [], [], []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        labels_numpy_all, pred_numpy_all = np.zeros(1), np.zeros(1)
        for data, labels in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)
            h, _ = model(data)
            fea_concat = h
            predictions_test = classifier(fea_concat)
            fea_concat_flat = fea_concat.reshape(fea_concat.shape[0], -1)
            loss = criterion(predictions_test, labels)
            acc_bs = labels.eq(predictions_test.detach().argmax(dim=1)).float().mean()
            total_acc.append(acc_bs.cpu().item())
            pred_numpy = predictions_test.detach().cpu().numpy()
            labels_numpy = labels.detach().cpu().numpy()
            try:
                auc_bs = roc_auc_score(F.one_hot(labels).detach().cpu().numpy(), pred_numpy, average="macro", multi_class="ovr")
            except Exception:
                auc_bs = 0.0
            try:
                prc_bs = average_precision_score(F.one_hot(labels).detach().cpu().numpy(), pred_numpy, average="macro")
            except Exception:
                prc_bs = 0.0
            if auc_bs != 0:
                total_auc.append(auc_bs)
            if prc_bs != 0:
                total_prc.append(prc_bs)
            pred_numpy = np.argmax(pred_numpy, axis=1)
            precision = precision_score(labels_numpy, pred_numpy, average='macro')
            recall = recall_score(labels_numpy, pred_numpy, average='macro')
            F1 = f1_score(labels_numpy, pred_numpy, average='macro')
            total_precision.append(precision)
            total_recall.append(recall)
            total_f1.append(F1)
            total_loss.append(loss.item())
            labels_numpy_all = np.concatenate((labels_numpy_all, labels_numpy))
            pred_numpy_all = np.concatenate((pred_numpy_all, pred_numpy))
    labels_numpy_all = labels_numpy_all[1:]
    pred_numpy_all = pred_numpy_all[1:]
    precision = precision_score(labels_numpy_all, pred_numpy_all, average='macro')
    recall = recall_score(labels_numpy_all, pred_numpy_all, average='macro')
    F1 = f1_score(labels_numpy_all, pred_numpy_all, average='macro')
    acc = accuracy_score(labels_numpy_all, pred_numpy_all)
    total_loss = float(torch.tensor(total_loss).mean()) if total_loss else 0.0
    total_acc = float(np.mean(total_acc)) if total_acc else 0.0
    total_auc = float(np.mean(total_auc)) if total_auc else 0.0
    total_prc = float(np.mean(total_prc)) if total_prc else 0.0
    performance = [acc * 100, precision * 100, recall * 100, F1 * 100, total_auc * 100, total_prc * 100]
    return total_loss, total_acc, total_auc, total_prc, performance
