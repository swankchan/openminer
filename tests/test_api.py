"""Test API endpoints."""
import time
import requests


def test_api():
    """Test API endpoints."""
    base_url = "http://localhost:8000"

    print("=" * 60)
    print("Testing API endpoints")
    print("=" * 60)

    # Wait for the server to start
    print("\nWaiting for the server to start...")
    time.sleep(2)

    # 1. Test homepage
    print("\n1. Test homepage (GET /):")
    try:
        response = requests.get(f"{base_url}/", timeout=5)
        if response.status_code == 200:
            print(f"   ✓ Homepage OK (status: {response.status_code})")
            print(f"   Content length: {len(response.text)} characters")
        else:
            print(f"   ✗ Homepage unexpected (status: {response.status_code})")
            return False
    except requests.exceptions.ConnectionError:
        print("   ✗ Cannot connect to server. Ensure the app is running.")
        print("   Start command: python run.py")
        return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

    # 2. Test download endpoint (should return 404 because there is no file)
    print("\n2. Test download endpoint (GET /api/download/test):")
    try:
        response = requests.get(f"{base_url}/api/download/test", timeout=5)
        if response.status_code == 404:
            print("   ✓ Download endpoint OK (correctly returned 404)")
        else:
            print(f"   ⚠ Download endpoint returned status: {response.status_code}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n" + "=" * 60)
    print("✓ API test completed!")
    print("=" * 60)
    print("\nThe application appears to be running. Next steps:")
    print("  1. Visit http://localhost:8000 to use the web UI")
    print("  2. Use POST /api/upload to upload a PDF")
    print("  3. Use POST /api/process-folder to process PDFs in a folder")

    return True


if __name__ == "__main__":
    test_api()
