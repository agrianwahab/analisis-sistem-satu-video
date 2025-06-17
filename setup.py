import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 7):
        print("Error: Python 3.7 or higher is required")
        sys.exit(1)

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Successfully installed required packages")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        sys.exit(1)

def verify_imports():
    """Verify that all required packages can be imported"""
    packages = [
        'numpy',
        'cv2',
        'torch',
        'tensorflow',
        'PIL',
        'imagehash',
        'skimage',
        'matplotlib',
        'tqdm'
    ]
    
    missing = []
    for package in packages:
        try:
            if package == 'cv2':
                import cv2
            elif package == 'PIL':
                from PIL import Image
            else:
                __import__(package)
            print(f"✓ Successfully imported {package}")
        except ImportError:
            missing.append(package)
            print(f"✗ Failed to import {package}")
    
    return missing

def main():
    """Main setup function"""
    print("Starting setup...")
    
    # Check Python version
    check_python_version()
    
    # Install requirements
    install_requirements()
    
    # Verify imports
    print("\nVerifying package imports...")
    missing = verify_imports()
    
    if missing:
        print(f"\nWarning: The following packages could not be imported: {', '.join(missing)}")
        print("Please try installing them manually or check for any error messages above")
    else:
        print("\nSetup completed successfully! All required packages are installed and working.")
        print("\nYou can now run the video forensics system using:")
        print("python enhanced_video_forensics.py")

if __name__ == "__main__":
    main()
