"""Script to check vLLM installation status."""
import sys

def check_vllm():
    """Check vLLM installation status."""
    print("=" * 60)
    print("Checking vLLM installation status")
    print("=" * 60)
    
    # 1. Check if vllm is installed
    print("\n1. Check vLLM installation:")
    try:
        import vllm
        print("   ✓ vllm is installed")
        if hasattr(vllm, '__version__'):
            print(f"   Version: {vllm.__version__}")
    except ImportError:
        print("   ✗ vllm is not installed")
        print("   Install command: pip install vllm")
        return
    
    # 2. Check AsyncLLM
    print("\n2. Check AsyncLLM:")
    try:
        from vllm.v1.engine.async_llm import AsyncLLM
        print("   ✓ AsyncLLM can be imported")
    except ImportError as e:
        print(f"   ✗ Failed to import AsyncLLM: {e}")
    
    # 3. Check AsyncEngineArgs
    print("\n3. Check AsyncEngineArgs:")
    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
        print("   ✓ AsyncEngineArgs can be imported")
        
        # Inspect parameters
        import inspect
        sig = inspect.signature(AsyncEngineArgs.__init__)
        params = list(sig.parameters.keys())
        print(f"   Available parameters: {', '.join(params[:10])}...")  # Only show the first 10
    except ImportError as e:
        print(f"   ✗ Failed to import AsyncEngineArgs: {e}")
    
    # 4. Check vllm._C module
    print("\n4. Check vllm._C module (C extension):")
    try:
        import vllm._C
        print("   ✓ vllm._C is available (C extension is built)")
    except ImportError as e:
        print(f"   ⚠ vllm._C is not available: {e}")
        print("   This may indicate vllm needs to be rebuilt or reinstalled")
    
    # 5. Check PyTorch
    print("\n5. Check PyTorch:")
    try:
        import torch
        print(f"   ✓ PyTorch version: {torch.__version__}")
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   CUDA version: {torch.version.cuda}")
            print(f"   GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("   ⚠ No CUDA GPU available; CPU will be used (slower)")
    except ImportError:
        print("   ✗ PyTorch is not installed")
    
    # 6. Check mineru_vl_utils
    print("\n6. Check mineru_vl_utils:")
    try:
        from mineru_vl_utils import MinerUClient
        print("   ✓ mineru_vl_utils is installed")
        
        try:
            from mineru_vl_utils import MinerULogitsProcessor
            print("   ✓ MinerULogitsProcessor is available (vllm >= 0.10.1)")
        except ImportError:
            print("   ⚠ MinerULogitsProcessor is not available (vllm may be < 0.10.1)")
    except ImportError as e:
        print(f"   ✗ mineru_vl_utils is not installed: {e}")
        print("   Install command: pip install mineru-vl-utils")
    
    # 7. Check environment variables
    print("\n7. Check related environment variables:")
    import os
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "not set")
    print(f"   CUDA_VISIBLE_DEVICES: {cuda_visible}")
    
    # 8. Summary
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    issues = []
    try:
        import vllm
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.engine.arg_utils import AsyncEngineArgs
        from mineru_vl_utils import MinerUClient
        print("✓ Core components are installed; vllm-async-engine should be usable")
    except ImportError as e:
        issues.append(f"Missing required component: {e}")
    
    try:
        import vllm._C
    except ImportError:
        issues.append("vllm._C is not available; it may need to be rebuilt")
    
    if issues:
        print("\n⚠ Issues detected:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nRecommendations:")
        print("  1. Reinstall vllm: pip uninstall vllm && pip install vllm")
        print("  2. If using GPU, ensure CUDA version compatibility")
        print("  3. If issues persist, you can temporarily use CLI mode (USE_VLLM_ASYNC=False)")
    else:
        print("\n✓ All checks passed; vllm-async-engine should work normally")

if __name__ == "__main__":
    check_vllm()

