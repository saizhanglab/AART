"""
Utility functions for saving and loading scalers, log transformation parameters, and feature names.
"""

import pickle
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def save_scalers_and_features(
    datamodule,
    output_dir: str,
    scalers_filename: str = 'scalers.pkl',
    features_filename: str = 'feature_names.pkl',
    log_params_filename: str = 'log_transform_params.pkl'
) -> Dict[str, str]:
    """
    Save fitted scalers, log transformation parameters, and feature names from a datamodule.
    
    Args:
        datamodule: JointVAEDataModule with fitted preprocessor
        output_dir: Directory to save files
        scalers_filename: Filename for scalers pickle file
        features_filename: Filename for feature names pickle file
        log_params_filename: Filename for log transformation parameters pickle file
        
    Returns:
        Dictionary with paths to saved files
    """
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get preprocessor
    preprocessor = datamodule.get_preprocessor()
    
    # Extract scalers, feature names, and log transformation parameters
    scalers = preprocessor.get_scalers()
    feature_names = preprocessor.get_feature_names()
    log_transform_params = preprocessor.get_log_transform_params()
    
    # Save scalers
    scalers_path = output_path / scalers_filename
    with open(scalers_path, 'wb') as f:
        pickle.dump(scalers, f)
    logger.info(f"Scalers saved to {scalers_path}")
    
    # Save feature names
    features_path = output_path / features_filename
    with open(features_path, 'wb') as f:
        pickle.dump(feature_names, f)
    logger.info(f"Feature names saved to {features_path}")
    
    # Save log transformation parameters
    log_params_path = output_path / log_params_filename
    with open(log_params_path, 'wb') as f:
        pickle.dump(log_transform_params, f)
    logger.info(f"Log transformation parameters saved to {log_params_path}")
    
    # Log information about what was saved
    logger.info("Scaler information:")
    for platform, scaler in scalers.items():
        if scaler is not None:
            logger.info(f"  {platform}: {type(scaler).__name__}")
            if hasattr(scaler, 'mean_'):
                logger.info(f"    Features: {len(scaler.mean_)}")
        else:
            logger.info(f"  {platform}: None")
    
    logger.info("Feature names information:")
    for platform, names in feature_names.items():
        if names is not None:
            logger.info(f"  {platform}: {len(names)} features")
        else:
            logger.info(f"  {platform}: None")
    
    logger.info("Log transformation information:")
    for platform, params in log_transform_params.items():
        if params['enabled']:
            logger.info(f"  {platform}: Enabled, shift={params['shift_value']:.6f}")
        else:
            logger.info(f"  {platform}: Disabled")
    
    return {
        'scalers_path': str(scalers_path),
        'features_path': str(features_path),
        'log_params_path': str(log_params_path)
    }


def load_scalers_and_features(
    experiment_dir: Optional[str] = None,
    model_path: Optional[str] = None,
    scalers_filename: str = 'scalers.pkl',
    features_filename: str = 'feature_names.pkl',
    log_params_filename: str = 'log_transform_params.pkl'
) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    """
    Load scalers, log transformation parameters, and feature names from a saved experiment directory.

    Args:
        experiment_dir: Path to the root of a saved experiment version.
        model_path: Path to model checkpoint (used to infer experiment_dir if not provided).
        scalers_filename: The filename for the scalers pickle file.
        features_filename: The filename for the feature names pickle file.
        log_params_filename: The filename for the log transformation parameters pickle file.

    Returns:
        Tuple of (scalers, feature_names, log_transform_params)
    """
    # Determine base directory
    if experiment_dir:
        base_dir = Path(experiment_dir)
    elif model_path:
        # Infer from model path, assuming it's in a checkpoint subdirectory
        base_dir = Path(model_path).parent.parent 
    else:
        raise ValueError("Must provide either 'experiment_dir' or 'model_path'.")

    scalers_path = base_dir / scalers_filename
    features_path = base_dir / features_filename
    log_params_path = base_dir / log_params_filename
    
    # Load scalers
    scalers = None
    if scalers_path.exists():
        try:
            with open(scalers_path, 'rb') as f:
                scalers = pickle.load(f)
            print(f"Loaded scalers from {scalers_path}")
        except Exception as e:
            print(f"Error loading scalers: {e}")
    else:
        print(f"Warning: Scalers not found at {scalers_path}")
        print("Make sure the experiment directory is correct and contains the scalers file.")
    
    # Load feature names
    feature_names = None
    if features_path.exists():
        try:
            with open(features_path, 'rb') as f:
                feature_names = pickle.load(f)
            print(f"Loaded feature names from {features_path}")
        except Exception as e:
            print(f"Error loading feature names: {e}")
    else:
        print(f"Warning: Feature names not found at {features_path}")
    
    # Load log transformation parameters
    log_transform_params = None
    if log_params_path.exists():
        try:
            with open(log_params_path, 'rb') as f:
                log_transform_params = pickle.load(f)
            print(f"Loaded log transformation parameters from {log_params_path}")
        except Exception as e:
            print(f"Error loading log transformation parameters: {e}")
    else:
        print(f"Warning: Log transformation parameters not found at {log_params_path}")
        # Create default parameters if not found (backward compatibility)
        log_transform_params = {
            'platform_a': {'enabled': False, 'shift_value': 0.0},
            'platform_b': {'enabled': False, 'shift_value': 0.0}
        }
        print("Using default log transformation parameters (no transformation)")
    
    return scalers, feature_names, log_transform_params


def validate_scalers_for_inference(scalers: Optional[Dict], platform: str) -> bool:
    """
    Validate that scalers are available for a given platform.
    
    Args:
        scalers: Dictionary of scalers (can be None)
        platform: Platform name (e.g., 'platform_a')
        
    Returns:
        True if scaler is available and valid
    """
    if scalers is None:
        return False
    
    if platform not in scalers:
        return False
    
    scaler = scalers[platform]
    if scaler is None:
        return False
    
    if not hasattr(scaler, 'transform'):
        return False
    
    return True


def preprocess_input_data(
    source_df: pd.DataFrame, 
    config: Dict, 
    scalers: Optional[Dict], 
    log_transform_params: Optional[Dict],
    source_platform: str
) -> Tuple[np.ndarray, str]:
    """
    Preprocess input data using the same pipeline as training.
    
    Args:
        source_df: Source data DataFrame
        config: Configuration dictionary
        scalers: Dictionary of fitted scalers
        log_transform_params: Dictionary of log transformation parameters
        source_platform: Source platform name
        
    Returns:
        Tuple of (normalized_features, id_column_name)
    """
    # Extract ID column and features
    id_column = source_df.columns[0]
    feature_columns = source_df.columns[1:]
    
    print(f"Found {len(feature_columns)} features for {len(source_df)} samples")
    
    # Extract features
    source_features = source_df[feature_columns]
    
    # Handle missing values using the same strategy as training
    missing_strategy = config['data']['missing_value_strategy']
    if source_features.isnull().any().any():
        print(f"Found missing values. Handling using strategy: {missing_strategy}")
        
        if missing_strategy == 'mean':
            source_features = source_features.fillna(source_features.mean())
        elif missing_strategy == 'median':
            source_features = source_features.fillna(source_features.median())
        elif missing_strategy == 'drop':
            raise ValueError("Cannot use 'drop' strategy for inference with missing values")
        elif missing_strategy == 'knn':
            # For KNN imputation during inference, fall back to mean imputation for simplicity
            print("Warning: KNN imputation not supported during inference, falling back to mean imputation")
            source_features = source_features.fillna(source_features.mean())
        else:
            # Default fallback for any other strategy
            print(f"Warning: Unknown missing value strategy '{missing_strategy}', falling back to mean imputation")
            source_features = source_features.fillna(source_features.mean())
    
    # Apply log transformation if enabled
    if log_transform_params and source_platform in log_transform_params:
        params = log_transform_params[source_platform]
        if params['enabled']:
            print(f"Applying log transformation to {source_platform}")
            shift_value = params['shift_value']
            shifted_data = source_features + shift_value
            source_features = pd.DataFrame(
                np.log(shifted_data), 
                columns=source_features.columns, 
                index=source_features.index
            )
    
    # Normalize data using fitted scalers
    if validate_scalers_for_inference(scalers, source_platform):
        scaler = scalers[source_platform]
        source_features_norm = scaler.transform(source_features)
        print(f"Applied {source_platform} scaler for normalization")
    else:
        print("Warning: No valid scaler available. Using raw data (this may lead to poor results!)")
        source_features_norm = source_features.values
    
    return source_features_norm.astype(np.float32), id_column


def inverse_transform_output(
    imputed_data: np.ndarray, 
    scalers: Optional[Dict], 
    log_transform_params: Optional[Dict],
    target_platform: str
) -> np.ndarray:
    """
    Transform imputed data back to original scale.
    
    Args:
        imputed_data: Imputed data in normalized space
        scalers: Dictionary of fitted scalers
        log_transform_params: Dictionary of log transformation parameters
        target_platform: Target platform name
        
    Returns:
        Data in original scale
    """
    # First inverse the normalization
    if validate_scalers_for_inference(scalers, target_platform):
        scaler = scalers[target_platform]
        imputed_data_unnorm = scaler.inverse_transform(imputed_data)
        print(f"Applied inverse normalization using {target_platform} scaler")
    else:
        print("Warning: No valid scaler available for inverse transformation")
        imputed_data_unnorm = imputed_data
    
    # Then inverse the log transformation if it was applied
    if log_transform_params and target_platform in log_transform_params:
        params = log_transform_params[target_platform]
        if params['enabled']:
            print(f"Applying inverse log transformation to {target_platform}")
            shift_value = params['shift_value']
            exp_data = np.exp(imputed_data_unnorm)
            imputed_data_original = exp_data - shift_value
            return imputed_data_original
    
    return imputed_data_unnorm 