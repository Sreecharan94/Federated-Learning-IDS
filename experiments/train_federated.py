# experiments/train_federated.py
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['KAFKA_BOOTSTRAP_SERVERS'] = 'disabled'

import logging
import yaml
import numpy as np
import tensorflow as tf
from preprocessing.pipeline import PreprocessingPipeline
from federated.federated_trainer import FederatedTrainer
from monitoring.metrics import MetricsCollector
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_federated_experiment(config_path: str = "./configs/config.yaml"):
    """Main federated training experiment"""
    logger.info("Starting federated training experiment...")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    metrics_collector = MetricsCollector(config_path)
    metrics_collector.start_collection()

    try:
        logger.info("Initializing preprocessing pipeline...")
        preprocessing_pipeline = PreprocessingPipeline(config_path)

        logger.info("Loading and preprocessing data (chunked)...")
        train_data, val_data, test_data, metadata = preprocessing_pipeline.preprocess_data()

        # CORRECTED: Access actual data arrays (not metadata)
        logger.info(f"Training set shape: {train_data['X'].shape}")
        logger.info(f"Validation set shape: {val_data['X'].shape}")
        logger.info(f"Test set shape: {test_data['X'].shape}")
        logger.info(f"Number of classes: {metadata['num_classes']}")

        logger.info("Initializing federated trainer...")
        federated_trainer = FederatedTrainer(config_path)
        federated_trainer.setup_federated_system(metadata['num_classes'])

        logger.info("Starting federated training...")
        training_results = federated_trainer.train_federated_model(
            train_data['X'],
            train_data['y'],
            test_data['X'],
            test_data['y']
        )

        logger.info("Evaluating final federated model...")
        final_eval = federated_trainer.evaluate_federated_model(
            test_data['X'],
            test_data['y']
        )

        logger.info(f"Final evaluation - Loss: {final_eval['global_loss']:.4f}, "
                   f"Accuracy: {final_eval['global_accuracy']:.4f}")

        logger.info("Generating training plots...")
        federated_trainer.plot_training_progress(
            save_path="./outputs/plots/federated_training_progress.png"
        )

        client_stats = federated_trainer.get_client_statistics()
        logger.info(f"Client statistics: {client_stats}")

        metrics_collector.record_federated_round(
            round_num=config['federated']['num_rounds'],
            accuracy=final_eval['global_accuracy'],
            loss=final_eval['global_loss'],
            active_clients=client_stats.get('total_clients', 0),
            round_time=0
        )

        metrics_collector.export_metrics("./outputs/experiment_metrics/")

        logger.info("Federated training experiment completed successfully!")

        return {
            'training_results': training_results,
            'final_evaluation': final_eval,
            'metadata': metadata,
            'client_stats': client_stats
        }

    except Exception as e:
        logger.error(f"Error in federated training experiment: {e}")
        raise
    finally:
        metrics_collector.stop_collection()

def main():
    parser = argparse.ArgumentParser(description='FL-IDS Federated Training Experiment')
    parser.add_argument('--config', type=str, default='./configs/config.yaml',
                       help='Path to configuration file')

    args = parser.parse_args()

    os.makedirs('./outputs/plots/', exist_ok=True)
    os.makedirs('./outputs/experiment_metrics/', exist_ok=True)
    os.makedirs('./models/', exist_ok=True)

    results = train_federated_experiment(args.config)

    print("\n" + "="*50)
    print("FEDERATED TRAINING EXPERIMENT COMPLETED")
    print("="*50)
    print(f"Final Accuracy: {results['final_evaluation']['global_accuracy']:.4f}")
    print(f"Final Loss: {results['final_evaluation']['global_loss']:.4f}")
    print(f"Number of Classes: {results['metadata']['num_classes']}")
    print(f"Number of Clients: {results['client_stats']['total_clients']}")
    print(f"Total Samples: {results['client_stats']['total_samples']}")
    print("="*50)

if __name__ == "__main__":
    main()