"""
Script to set up the environment for Arabic Image Captioning.
"""

import subprocess
import sys
import os
from pathlib import Path

def install_dependencies():
    """Install required dependencies."""
    print("Installing required dependencies...")
    
    # Core dependencies
    dependencies = [
        "torch",
        "torchvision",
        "pillow",
        "numpy",
        "tqdm",
        "fastapi",
        "uvicorn",
        "python-multipart"
    ]
    
    # Install each dependency
    for dep in dependencies:
        print(f"Installing {dep}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
    
    print("All dependencies installed successfully!")

def check_directories():
    """Check if required directories exist and create them if needed."""
    print("Checking required directories...")
    
    # Define required directories
    base_dir = Path(__file__).resolve().parent
    directories = [
        base_dir / "models",
        base_dir / "models" / "checkpoints",
        base_dir / "logs"
    ]
    
    # Create directories if they don't exist
    for directory in directories:
        if not directory.exists():
            print(f"Creating directory: {directory}")
            os.makedirs(directory, exist_ok=True)
    
    print("All required directories are set up!")

def main():
    """Main function to set up the environment."""
    print("Setting up environment for Arabic Image Captioning...")
    
    # Install dependencies
    install_dependencies()
    
    # Check directories
    check_directories()
    
    print("\nEnvironment setup complete!")
    print("You can now run the model with 'python test_model.py'")

if __name__ == "__main__":
    main()
