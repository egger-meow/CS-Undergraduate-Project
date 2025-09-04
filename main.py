# main.py - Main execution script for the refactored dual-AE system
import argparse
import logging
import sys
import os
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from core.data_processor import DataProcessor
from core.dual_autoencoder import DualAutoEncoder
from core.fusion_engine import FusionEngine
from core.evaluation_pipeline import EvaluationPipeline

def setup_logging():
    """Setup logging configuration"""
    log_dir = os.path.join(config.system.project_root, config.system.logs_dir)
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'dual_ae_pipeline_{timestamp}.log')
    
    logging.basicConfig(
        level=getattr(logging, config.system.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_file}")
    return logger

def run_single_mode_training(mode: str):
    """Run training for a single mode (amp or vib)"""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting single mode training: {mode}")
    
    # Initialize components
    data_processor = DataProcessor()
    
    # Prepare data based on mode
    if mode == 'amp':
        dataloaders = data_processor.prepare_amp_data()
    elif mode == 'vib':
        dataloaders = data_processor.prepare_vib_data()
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    # Train dual autoencoder
    dual_ae = DualAutoEncoder(mode=mode)
    dual_ae.train_dual_system(dataloaders)
    
    # Save models
    model_paths = config.get_model_paths(mode)
    dual_ae.save_models(model_paths)
    
    # Extract features
    features, labels = dual_ae.extract_features(dataloaders)
    
    # Train classifiers
    fusion_engine = FusionEngine()
    trained_classifiers = {}
    
    for clf_name in config.evaluation.classifiers:
        clf = fusion_engine.train_classifier(features, labels, clf_name)
        trained_classifiers[clf_name] = clf
    
    # Save classifiers
    if mode == 'amp':
        fusion_engine.save_pipelines(trained_classifiers, {})
    else:
        fusion_engine.save_pipelines({}, trained_classifiers)
    
    logger.info(f"Single mode training completed: {mode}")

def run_dual_mode_evaluation():
    """Run dual mode evaluation with fusion"""
    logger = logging.getLogger(__name__)
    logger.info("Starting dual mode evaluation with fusion")
    
    # Initialize components
    data_processor = DataProcessor()
    fusion_engine = FusionEngine()
    
    # Prepare data for both modes
    amp_dataloaders = data_processor.prepare_amp_data()
    vib_dataloaders = data_processor.prepare_vib_data()
    
    # Load or train amp models
    amp_dual_ae = DualAutoEncoder(mode='amp')
    amp_model_paths = config.get_model_paths('amp')
    
    if all(os.path.exists(path) for path in amp_model_paths.values()):
        logger.info("Loading existing amp models")
        amp_dual_ae.load_models(amp_model_paths)
    else:
        logger.info("Training new amp models")
        amp_dual_ae.train_dual_system(amp_dataloaders)
        amp_dual_ae.save_models(amp_model_paths)
    
    # Load or train vib models
    vib_dual_ae = DualAutoEncoder(mode='vib')
    vib_model_paths = config.get_model_paths('vib')
    
    if all(os.path.exists(path) for path in vib_model_paths.values()):
        logger.info("Loading existing vib models")
        vib_dual_ae.load_models(vib_model_paths)
    else:
        logger.info("Training new vib models")
        vib_dual_ae.train_dual_system(vib_dataloaders)
        vib_dual_ae.save_models(vib_model_paths)
    
    # Extract features
    amp_features, amp_labels = amp_dual_ae.extract_features(amp_dataloaders)
    vib_features, vib_labels = vib_dual_ae.extract_features(vib_dataloaders)
    
    # Run fusion evaluation
    results = fusion_engine.train_and_evaluate_pipeline(
        amp_features, amp_labels,
        vib_features, vib_labels,
        amp_features, amp_labels,
        vib_features, vib_labels
    )
    
    # Create and display results summary
    summary = fusion_engine.create_results_summary(results)
    
    logger.info("Dual mode evaluation completed")
    print("\n=== RESULTS SUMMARY ===")
    print(summary.to_string(index=False))
    
    return results, summary

def run_complete_evaluation():
    """Run the complete 10-repeat evaluation pipeline"""
    logger = logging.getLogger(__name__)
    logger.info("Starting complete 10-repeat evaluation pipeline")
    
    evaluation_pipeline = EvaluationPipeline()
    final_report = evaluation_pipeline.run_complete_evaluation()
    
    logger.info("Complete evaluation pipeline finished")
    return final_report

def main():
    """Main entry point with command line interface"""
    parser = argparse.ArgumentParser(
        description="Elevator Re-leveling Anomaly Detection - Dual Autoencoder System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --train-amp          # Train amplitude model only
  python main.py --train-vib          # Train vibration model only  
  python main.py --train-both         # Train both models
  python main.py --evaluate           # Evaluate with fusion
  python main.py --complete           # Run complete 10-repeat evaluation
  python main.py --train-both --evaluate  # Train and evaluate
        """
    )
    
    parser.add_argument('--train-amp', action='store_true',
                       help='Train amplitude (motor current) model')
    parser.add_argument('--train-vib', action='store_true', 
                       help='Train vibration (door XYZ) model')
    parser.add_argument('--train-both', action='store_true',
                       help='Train both amp and vib models')
    parser.add_argument('--evaluate', action='store_true',
                       help='Run dual-mode evaluation with fusion')
    parser.add_argument('--complete', action='store_true',
                       help='Run complete 10-repeat evaluation pipeline')
    parser.add_argument('--config-info', action='store_true',
                       help='Display current configuration')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    try:
        # Display configuration if requested
        if args.config_info:
            print("=== CURRENT CONFIGURATION ===")
            print(f"Data version: {config.data.data_version}")
            print(f"Architecture: {config.model.architecture}")
            print(f"Amp channels: {config.data.amp_channels}")
            print(f"Vib channels: {config.data.vib_channels}")
            print(f"Embedding sizes: amp={config.model.embedding_size}, vib={config.model.embedding_size_vib}")
            print(f"Device: {config.system.device}")
            print(f"Fusion method: {config.evaluation.fusion_method}")
            print(f"N repeats: {config.evaluation.n_repeats}")
            return
        
        # Check if any action is specified
        if not any([args.train_amp, args.train_vib, args.train_both, 
                   args.evaluate, args.complete]):
            parser.print_help()
            return
        
        # Train amplitude model
        if args.train_amp or args.train_both:
            run_single_mode_training('amp')
        
        # Train vibration model  
        if args.train_vib or args.train_both:
            run_single_mode_training('vib')
        
        # Run evaluation
        if args.evaluate:
            run_dual_mode_evaluation()
        
        # Run complete evaluation
        if args.complete:
            run_complete_evaluation()
        
        logger.info("Main execution completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        raise

if __name__ == "__main__":
    main()
