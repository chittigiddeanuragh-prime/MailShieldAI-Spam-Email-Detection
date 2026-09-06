import sys
import os

# Set stdout UTF-8 encoding for Windows compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add root project path to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.pipeline.train_pipeline import TrainPipeline

if __name__ == "__main__":
    print("[MailShield AI] Starting Model Training...")
    trainer = TrainPipeline(model_dir="models")
    metrics = trainer.run_training()
    print("[MailShield AI] Model Training Complete!")
