#!/usr/bin/env python3
"""
Download Trained Models from Google Colab

This script helps you download models trained on Google Colab.
Supports Google Drive direct downloads.
"""

import os
import sys
import argparse
from pathlib import Path

def print_instructions():
    """Print download instructions."""
    print("\n" + "="*80)
    print("📥 DOWNLOAD MODELS FROM GOOGLE COLAB")
    print("="*80 + "\n")
    
    print("After training on Google Colab, your models are saved to:")
    print("  📂 /MyDrive/autonomous_colony_models/\n")
    
    print("📖 Download Methods:\n")
    
    print("Method 1: Direct Download from Google Drive (Recommended)")
    print("  1. Open Google Drive in your browser")
    print("  2. Navigate to 'My Drive/autonomous_colony_models'")
    print("  3. Select the model files (.pt files)")
    print("  4. Right-click → Download")
    print("  5. Move to your local 'models/' directory\n")
    
    print("Method 2: Download Zip (All Models at Once)")
    print("  1. In Colab, run the 'Zip Models' cell (last cell)")
    print("  2. The zip file will download automatically")
    print("  3. Extract to your local 'models/' directory\n")
    
    print("Method 3: Google Drive Desktop App")
    print("  1. Install Google Drive for Desktop")
    print("  2. Models sync automatically to your computer")
    print("  3. Navigate to: ~/Google Drive/My Drive/autonomous_colony_models/")
    print("  4. Copy models to this project's models/ directory\n")
    
    print("Method 4: gdown (Command Line)")
    print("  1. Install: pip install gdown")
    print("  2. Get shareable link from Google Drive")
    print("  3. Run: gdown <GOOGLE_DRIVE_LINK>")
    print("  4. Move downloaded file to models/\n")
    
    print("="*80)
    print("💡 TIP: Use Method 2 (Zip) if you trained multiple models!")
    print("="*80 + "\n")


def verify_model(model_path: str):
    """Verify a downloaded model file."""
    import torch
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        print(f"\n✅ Model Verified: {model_path}")
        print(f"   Episode: {checkpoint.get('episode', 'Unknown')}")
        
        if 'config' in checkpoint:
            config = checkpoint['config']
            print(f"   Agent Type: {config.get('agent_type', 'Unknown')}")
            print(f"   Training Episodes: {config.get('n_episodes', 'Unknown')}")
        
        if 'final_stats' in checkpoint:
            stats = checkpoint['final_stats']
            print(f"   Avg Reward: {stats.get('avg_reward', 'Unknown'):.2f}")
            print(f"   Success Rate: {stats.get('success_rate', 'Unknown'):.1%}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Model Verification Failed: {model_path}")
        print(f"   Error: {str(e)}")
        return False


def list_models():
    """List all models in the models directory."""
    models_dir = Path('models')
    
    if not models_dir.exists():
        print(f"\n⚠️  Models directory not found: {models_dir}")
        print("   Creating directory...")
        models_dir.mkdir(exist_ok=True)
        print(f"   ✓ Created: {models_dir}")
        return
    
    models = list(models_dir.glob('*.pt'))
    
    if not models:
        print(f"\n📁 No models found in {models_dir}/")
        print("   Download models from Google Colab first!")
        return
    
    print(f"\n📁 Models in {models_dir}/ ({len(models)}):")
    print("="*80)
    
    for i, model in enumerate(sorted(models), 1):
        size = model.stat().st_size / (1024 * 1024)  # MB
        print(f"{i}. {model.name} ({size:.2f} MB)")
    
    print("="*80 + "\n")


def setup_models_directory():
    """Ensure models directory exists."""
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    return models_dir


def main():
    parser = argparse.ArgumentParser(
        description="Download and verify models from Google Colab"
    )
    
    parser.add_argument(
        '--verify',
        type=str,
        help='Verify a specific model file'
    )
    
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all models in models/ directory'
    )
    
    parser.add_argument(
        '--setup',
        action='store_true',
        help='Setup models directory'
    )
    
    args = parser.parse_args()
    
    # Default: show instructions
    if not any([args.verify, args.list, args.setup]):
        print_instructions()
        list_models()
        return
    
    if args.setup:
        models_dir = setup_models_directory()
        print(f"✓ Models directory ready: {models_dir}/")
        return
    
    if args.list:
        list_models()
        return
    
    if args.verify:
        if not os.path.exists(args.verify):
            print(f"\n❌ Model file not found: {args.verify}")
            return
        
        verify_model(args.verify)


if __name__ == "__main__":
    main()
