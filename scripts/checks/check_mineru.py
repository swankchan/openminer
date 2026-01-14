"""Script to check MinerU installation status."""
import subprocess
import sys

def check_mineru():
    """Check whether MinerU is installed."""
    print("Checking MinerU installation status...\n")
    
    # Method 1: check Python module
    print("1. Check Python module:")
    try:
        import mineru
        print("   ✓ MinerU Python module is installed")
        if hasattr(mineru, '__version__'):
            print(f"   Version: {mineru.__version__}")
    except ImportError:
        print("   ✗ MinerU Python module is not installed")
    
    # Method 2: check CLI tool
    print("\n2. Check CLI tool:")
    try:
        result = subprocess.run(
            ["mineru", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("   ✓ MinerU CLI is available")
            print(f"   Output: {result.stdout.strip()}")
        else:
            print("   ✗ MinerU CLI execution failed")
    except FileNotFoundError:
        print("   ✗ MinerU CLI not found (not in PATH)")
    except Exception as e:
        print(f"   ✗ Error while checking CLI: {e}")
    
    # Method 3: check pip installation
    print("\n3. Check pip installation:")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", "mineru"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print("   ✓ MinerU is installed via pip")
            # Parse version information
            for line in result.stdout.split('\n'):
                if 'Version:' in line:
                    print(f"   {line.strip()}")
        else:
            print("   ✗ MinerU is not installed via pip")
    except Exception as e:
        print(f"   ✗ Error while checking pip: {e}")
    
    print("\n" + "="*50)
    print("Recommendation:")
    print("If MinerU is not installed, run: pip install mineru")
    print("Or visit the MinerU GitHub repository for installation instructions")

if __name__ == "__main__":
    check_mineru()

