#!/usr/bin/env python3

import logging
import os
import pickle
from pathlib import Path
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
import click
import tqdm

# Import PatchCore modules
import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils

LOGGER = logging.getLogger(__name__)


def load_patchcore_model(model_path, device):
    """Load a single PatchCore model from given path."""
    try:
        # Check if model path exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        # Load parameters
        params_file = os.path.join(model_path, "patchcore_params.pkl")
        if not os.path.exists(params_file):
            raise FileNotFoundError(f"Parameters file not found: {params_file}")
            
        with open(params_file, "rb") as f:
            patchcore_params = pickle.load(f)
        
        LOGGER.info(f"Loading model from: {model_path}")
        LOGGER.info(f"Model parameters: {patchcore_params}")
        
        # Initialize backbone
        backbone_name = patchcore_params["backbone.name"]
        if backbone_name == "wideresnet50":
            backbone = patchcore.backbones.load("wideresnet50")
            backbone.name = "wideresnet50"
        elif backbone_name == "wideresnet101":
            backbone = patchcore.backbones.load("wideresnet101")
            backbone.name = "wideresnet101"
        elif backbone_name == "resnext101":
            backbone = patchcore.backbones.load("resnext101")
            backbone.name = "resnext101"
        elif backbone_name == "densenet201":
            backbone = patchcore.backbones.load("densenet201")
            backbone.name = "densenet201"
        else:
            raise ValueError(f"Unknown backbone: {backbone_name}")
        
        # Initialize PatchCore
        loaded_patchcore = patchcore.patchcore.PatchCore(device)
        loaded_patchcore.load(
            backbone=backbone,
            layers_to_extract_from=patchcore_params["layers_to_extract_from"],
            device=device,
            input_shape=patchcore_params["input_shape"],
            pretrain_embed_dimension=patchcore_params["pretrain_embed_dimension"],
            target_embed_dimension=patchcore_params["target_embed_dimension"],
            patchsize=patchcore_params["patchsize"],
            patchstride=patchcore_params["patchstride"],
            anomaly_score_num_nn=patchcore_params["anomaly_scorer_num_nn"],
        )
        
        # Load the trained anomaly scorer
        loaded_patchcore.anomaly_scorer.load(model_path)
        
        LOGGER.info(f"Successfully loaded model: {model_path}")
        return loaded_patchcore
        
    except Exception as e:
        LOGGER.error(f"Failed to load model from {model_path}: {str(e)}")
        raise


def load_ensemble_models(model_paths, device):
    """Load ensemble of PatchCore models."""
    models = []
    for model_path in model_paths:
        model = load_patchcore_model(model_path, device)
        models.append(model)
    return models


def create_anomaly_mask(anomaly_map, threshold_config):
    """Create binary anomaly mask from anomaly score map with configurable threshold.
    
    Args:
        anomaly_map: Anomaly score map (2D numpy array)
        threshold_config: Dictionary containing 'threshold' and 'threshold_mode'
        
    Returns:
        Binary mask (0-255 values) where 255 indicates anomaly regions
    """
    threshold_value = threshold_config.get('threshold', None)
    threshold_mode = threshold_config.get('threshold_mode', 'adaptive')
    
    if threshold_mode == 'fixed' and threshold_value is not None:
        # Use fixed threshold value
        threshold = threshold_value
        
    elif threshold_mode == 'percentile' and threshold_value is not None:
        # Use percentile-based threshold
        threshold = np.percentile(anomaly_map, threshold_value)
        
    elif threshold_mode == 'std_multiplier' and threshold_value is not None:
        # Use mean + threshold_value * std
        threshold = np.mean(anomaly_map) + threshold_value * np.std(anomaly_map)
        
    else:
        # Default adaptive threshold: mean + 2 * std
        threshold = np.mean(anomaly_map) + 2 * np.std(anomaly_map)
    
    # Log threshold information for debugging
    LOGGER.debug(f"Threshold mode: {threshold_mode}, Value: {threshold_value}, Final threshold: {threshold:.4f}")
    LOGGER.debug(f"Anomaly map stats - Min: {anomaly_map.min():.4f}, Max: {anomaly_map.max():.4f}, Mean: {np.mean(anomaly_map):.4f}, Std: {np.std(anomaly_map):.4f}")
    
    mask = (anomaly_map > threshold).astype(np.uint8) * 255
    
    # Log mask statistics
    anomaly_pixels = np.sum(mask > 0)
    total_pixels = mask.size
    anomaly_ratio = anomaly_pixels / total_pixels * 100
    LOGGER.debug(f"Anomaly pixels: {anomaly_pixels}/{total_pixels} ({anomaly_ratio:.2f}%)")
    
    return mask


def create_mask_overlay(image, binary_mask, mask_color=(0, 0, 255), alpha=0.4):
    """Create visualization by overlaying binary mask on original image.
    
    Args:
        image: Original RGB image (numpy array)
        binary_mask: Binary mask (0-255 values)
        mask_color: Color for anomaly regions in BGR format (default: red)
        alpha: Transparency of the mask overlay
    """
    # Convert image to BGR for OpenCV
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image.copy()
    
    # Create colored mask
    colored_mask = np.zeros_like(image_bgr)
    
    # Apply mask color where mask is white (anomaly regions)
    mask_indices = binary_mask > 127  # Threshold for binary mask
    colored_mask[mask_indices] = mask_color
    
    # Blend original image with colored mask
    overlay = cv2.addWeighted(image_bgr, 1-alpha, colored_mask, alpha, 0)
    
    return overlay


def visualize_anomaly(image, anomaly_map, alpha=0.4):
    """Create visualization by overlaying anomaly map on original image."""
    # Normalize anomaly map to [0, 1]
    anomaly_map_normalized = (anomaly_map - anomaly_map.min()) / (anomaly_map.max() - anomaly_map.min() + 1e-8)
    
    # Create colored anomaly map (red heatmap)
    colored_anomaly = cv2.applyColorMap((anomaly_map_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Convert image to BGR if needed
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image
    
    # Blend images
    blended = cv2.addWeighted(image_bgr, 1-alpha, colored_anomaly, alpha, 0)
    
    return blended


def process_single_image(models, image_path, output_dir, dataset_name, data_path, resize_size, input_size, device, threshold_config):
    """Process a single image and save visualization results."""
    try:
        # Load and preprocess image
        image = Image.open(image_path).convert("RGB")
        original_size = image.size
        
        # Define transforms to match PatchCore preprocessing
        transform = transforms.Compose([
            transforms.Resize((resize_size, resize_size)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Transform image for model input
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        # Get anomaly maps from all models
        anomaly_maps = []
        anomaly_scores = []
        
        with torch.no_grad():
            for model in models:
                # Use the PatchCore predict method
                scores, segmentations = model.predict(input_tensor)[:2]
                
                # Extract segmentation map and scores
                if isinstance(segmentations, list):
                    segmentation = segmentations[0]
                else:
                    segmentation = segmentations
                
                if isinstance(scores, list):
                    score = scores[0]
                else:
                    score = scores
                
                # Convert to numpy if needed
                if torch.is_tensor(segmentation):
                    segmentation = segmentation.cpu().numpy()
                if torch.is_tensor(score):
                    score = score.cpu().numpy()
                
                anomaly_maps.append(segmentation)
                anomaly_scores.append(score)
        
        # Average anomaly maps if ensemble
        if len(anomaly_maps) > 1:
            final_anomaly_map = np.mean(anomaly_maps, axis=0)
            final_score = np.mean(anomaly_scores)
        else:
            final_anomaly_map = anomaly_maps[0]
            final_score = anomaly_scores[0]
        
        # Ensure proper shape for resizing
        if len(final_anomaly_map.shape) > 2:
            final_anomaly_map = final_anomaly_map.squeeze()
        
        # Resize anomaly map to original image size
        final_anomaly_map_resized = cv2.resize(
            final_anomaly_map, 
            original_size, 
            interpolation=cv2.INTER_LINEAR
        )
        
        # Create anomaly mask with custom threshold
        anomaly_mask = create_anomaly_mask(final_anomaly_map_resized, threshold_config)
        
        # Prepare original image for visualization
        original_image_np = np.array(image)
        
        # Create mask overlay visualization
        mask_overlay = create_mask_overlay(original_image_np, anomaly_mask)
        
        # Extract relative path structure from original image path
        # Get the dataset base path
        dataset_base_path = os.path.join(data_path, dataset_name)
        
        # Get relative path from dataset base to image file
        try:
            rel_path = os.path.relpath(image_path, dataset_base_path)
            # rel_path will be something like "test/broken_large/image.png"
            rel_dir = os.path.dirname(rel_path)  # "test/broken_large"
            image_filename = os.path.basename(rel_path)  # "image.png"
        except ValueError:
            # Fallback if relpath fails
            rel_dir = ""
            image_filename = os.path.basename(image_path)
        
        image_name = os.path.splitext(image_filename)[0]
        
        # Create output subdirectory preserving the original structure
        if rel_dir:
            dataset_output_dir = os.path.join(output_dir, dataset_name, rel_dir)
        else:
            dataset_output_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        # Save anomaly mask (binary)
        mask_path = os.path.join(dataset_output_dir, f"{image_name}_mask.png")
        cv2.imwrite(mask_path, anomaly_mask)
        
        # Save mask overlay visualization (binary mask on original image)
        vis_path = os.path.join(dataset_output_dir, f"{image_name}_visualization.png")
        cv2.imwrite(vis_path, mask_overlay)
        
        # Save anomaly heatmap
        heatmap_normalized = ((final_anomaly_map_resized - final_anomaly_map_resized.min()) / 
                             (final_anomaly_map_resized.max() - final_anomaly_map_resized.min() + 1e-8) * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
        heatmap_path = os.path.join(dataset_output_dir, f"{image_name}_heatmap.png")
        cv2.imwrite(heatmap_path, heatmap)
        
        return True, final_score
        
    except Exception as e:
        LOGGER.error(f"Error processing image {image_path}: {str(e)}")
        return False, None


def get_test_images(data_path, dataset_name):
    """Get all test images for a dataset."""
    test_dir = os.path.join(data_path, dataset_name, "test")
    
    image_paths = []
    if os.path.exists(test_dir):
        # Get images from all subdirectories (good, defect categories)
        for subdir in os.listdir(test_dir):
            subdir_path = os.path.join(test_dir, subdir)
            if os.path.isdir(subdir_path):
                for filename in os.listdir(subdir_path):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        image_paths.append(os.path.join(subdir_path, filename))
    
    return sorted(image_paths)


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--threshold", type=float, default=None, help="Custom threshold for anomaly mask generation. If not set, uses adaptive threshold.")
@click.option("--threshold-mode", type=click.Choice(['fixed', 'adaptive', 'percentile', 'std_multiplier']), default='adaptive', 
              help="Threshold calculation mode: fixed (use threshold as-is), adaptive (mean+2*std), percentile (threshold as percentile), std_multiplier (mean+threshold*std)")
def main(results_path, gpu, seed, threshold, threshold_mode):
    """PatchCore visualization script."""
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Starting PatchCore visualization...")
    
    # Log threshold configuration
    if threshold is not None:
        LOGGER.info(f"Using custom threshold: {threshold} (mode: {threshold_mode})")
    else:
        LOGGER.info(f"Using {threshold_mode} threshold mode")
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set device
    if torch.cuda.is_available() and gpu:
        device = torch.device(f"cuda:{gpu[0]}")
    else:
        device = torch.device("cpu")
    
    LOGGER.info(f"Using device: {device}")
    
    # Create results directory
    os.makedirs(results_path, exist_ok=True)


@main.result_callback()
def run(
    ctx_obj_list,
    results_path,
    gpu,
    seed,
    threshold,
    threshold_mode,
):
    """Execute visualization with loaded models and datasets."""
    device = torch.device(f"cuda:{gpu[0]}" if torch.cuda.is_available() and gpu else "cpu")
    
    # Store threshold parameters for use in processing
    threshold_config = {
        'threshold': threshold,
        'threshold_mode': threshold_mode
    }
    
    # Parse context objects
    patch_core_loader_context = None
    dataset_context = None
    
    for ctx_obj in ctx_obj_list:
        if isinstance(ctx_obj, dict):
            if "model_paths" in ctx_obj:
                patch_core_loader_context = ctx_obj
            elif "datasets" in ctx_obj:
                dataset_context = ctx_obj
    
    if patch_core_loader_context is None or dataset_context is None:
        raise click.ClickException("Both patch_core_loader and dataset contexts are required")
    
    model_paths = patch_core_loader_context["model_paths"]
    datasets = dataset_context["datasets"]
    data_path = dataset_context["data_path"]
    resize_size = dataset_context["resize"]
    input_size = dataset_context["imagesize"]
    
    LOGGER.info(f"Processing {len(datasets)} datasets with {len(model_paths)} models")
    
    # Process each dataset
    for i, dataset_name in enumerate(datasets):
        LOGGER.info(f"Processing dataset: {dataset_name} ({i+1}/{len(datasets)})")
        
        # Load corresponding model(s)
        if len(model_paths) == len(datasets):
            # One model per dataset
            current_models = [load_patchcore_model(model_paths[i], device)]
        elif len(model_paths) == 1:
            # Single model for all datasets
            current_models = [load_patchcore_model(model_paths[0], device)]
        else:
            # Ensemble models for each dataset
            models_per_dataset = len(model_paths) // len(datasets)
            start_idx = i * models_per_dataset
            end_idx = start_idx + models_per_dataset
            current_model_paths = model_paths[start_idx:end_idx]
            current_models = [load_patchcore_model(path, device) for path in current_model_paths]
        
        # Get test images
        test_images = get_test_images(data_path, dataset_name)
        LOGGER.info(f"Found {len(test_images)} test images for {dataset_name}")
        
        # Process each image
        successful_processed = 0
        failed_processed = 0
        
        for image_path in tqdm.tqdm(test_images, desc=f"Processing {dataset_name}"):
            success, score = process_single_image(
                current_models, 
                image_path, 
                results_path, 
                dataset_name, 
                data_path,
                resize_size, 
                input_size, 
                device,
                threshold_config
            )
            
            if success:
                successful_processed += 1
            else:
                failed_processed += 1
        
        LOGGER.info(f"Dataset {dataset_name}: {successful_processed} images processed successfully, {failed_processed} failed")
    
    LOGGER.info(f"Visualization completed. Results saved to: {results_path}")
    LOGGER.info("Output structure (preserving original test subdirectories):")
    LOGGER.info(f"  {results_path}/")
    for dataset_name in datasets:
        LOGGER.info(f"    ├── {dataset_name}/")
        LOGGER.info(f"    │   ├── test/")
        LOGGER.info(f"    │   │   ├── good/")
        LOGGER.info(f"    │   │   │   ├── *_mask.png          # Binary anomaly masks")
        LOGGER.info(f"    │   │   │   ├── *_visualization.png # Binary mask overlays")
        LOGGER.info(f"    │   │   │   └── *_heatmap.png       # Anomaly heatmaps")
        LOGGER.info(f"    │   │   └── defect_type/")
        LOGGER.info(f"    │   │       ├── *_mask.png")
        LOGGER.info(f"    │   │       ├── *_visualization.png")
        LOGGER.info(f"    │   │       └── *_heatmap.png")


@main.command("patch_core_loader")
@click.option("-p", "--model_path", type=str, multiple=True, required=True)
@click.option("--faiss_on_gpu", is_flag=True)
def patch_core_loader(model_path, faiss_on_gpu):
    """Load PatchCore model(s)."""
    return {
        "model_paths": list(model_path),
        "faiss_on_gpu": faiss_on_gpu
    }


@main.command("dataset")
@click.option("--resize", type=int, default=366, show_default=True)
@click.option("--imagesize", type=int, default=320, show_default=True)
@click.option("-d", "--dataset", type=str, multiple=True, required=True)
@click.argument("data_source", type=str)
@click.argument("data_path", type=str)
def dataset(resize, imagesize, dataset, data_source, data_path):
    """Configure dataset parameters."""
    return {
        "resize": resize,
        "imagesize": imagesize,
        "datasets": list(dataset),
        "data_source": data_source,
        "data_path": data_path
    }


if __name__ == "__main__":
    main()