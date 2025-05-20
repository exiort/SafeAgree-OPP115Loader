import sys
import subprocess
import importlib

def run_check(command_parts):
    """Runs a subprocess command and returns its output or error."""
    try:
        process = subprocess.run(command_parts, capture_output=True, text=True, check=False, timeout=60) # check=False to handle non-zero exits manually
        output = process.stdout.strip()
        error_output = process.stderr.strip()
        if process.returncode != 0:
            return output + "\n" + error_output, f"Command '{' '.join(command_parts)}' failed with exit code {process.returncode}."
        return output, None # No error string if successful
    except subprocess.TimeoutExpired:
        return None, f"Command '{' '.join(command_parts)}' timed out."
    except FileNotFoundError:
        return None, f"Command '{command_parts[0]}' not found. Is it in your PATH?"
    except Exception as e:
        return None, f"An unexpected error occurred with '{' '.join(command_parts)}': {str(e)}"

def check_library_import_and_version(library_name, version_attribute="__version__"):
    """Tries to import a library and get its version."""
    try:
        lib = importlib.import_module(library_name)
        version = getattr(lib, version_attribute, "N/A")
        return True, version, None
    except ImportError:
        return False, "N/A", f"Failed to import '{library_name}'."
    except Exception as e:
        return False, "N/A", f"Error importing or getting version for '{library_name}': {str(e)}"

print("Starting comprehensive environment validation...\n")
all_checks_passed = True
failed_checks_summary = []

# --- 1. PyTorch and CUDA ---
print("--- 1. PyTorch and CUDA Check ---")
pytorch_ok, pytorch_version, pytorch_error = check_library_import_and_version("torch")
if pytorch_ok:
    print(f"[PASS] PyTorch imported successfully. Version: {pytorch_version}")
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"   CUDA available: {cuda_available}")
        if cuda_available:
            print(f"   CUDA version detected by PyTorch: {torch.version.cuda}")
            print(f"   Number of GPUs available: {torch.cuda.device_count()}")
            if torch.cuda.device_count() > 0:
                print(f"   GPU name (GPU 0): {torch.cuda.get_device_name(0)}")
            else:
                print("[FAIL] CUDA is available but no GPUs were detected.")
                all_checks_passed = False
                failed_checks_summary.append("PyTorch: CUDA available but no GPUs detected.")
        else:
            print("[FAIL] CUDA is NOT available according to PyTorch.")
            all_checks_passed = False
            failed_checks_summary.append("PyTorch: CUDA not available.")
    except Exception as e:
        print(f"[FAIL] Error during PyTorch CUDA checks: {e}")
        all_checks_passed = False
        failed_checks_summary.append(f"PyTorch CUDA checks: {str(e)}")
else:
    print(f"[FAIL] {pytorch_error}")
    all_checks_passed = False
    failed_checks_summary.append(pytorch_error)
print("-" * 30 + "\n")

# --- 2. bitsandbytes Check ---
print("--- 2. bitsandbytes Check ---")
bnb_ok, bnb_version, bnb_error = check_library_import_and_version("bitsandbytes")
if bnb_ok:
    print(f"[PASS] bitsandbytes imported successfully. Version: {bnb_version}")
    try:
        import bitsandbytes as bnb
        if hasattr(bnb, 'COMPILED_WITH_CUDA'):
            print(f"   bitsandbytes.COMPILED_WITH_CUDA attribute: {bnb.COMPILED_WITH_CUDA}")
        else:
            print("   bitsandbytes.COMPILED_WITH_CUDA attribute not found (might be okay).")

        print("\n   Running 'python -m bitsandbytes' for detailed status:")
        bnb_cli_output, bnb_cli_error_str = run_check([sys.executable, "-m", "bitsandbytes"])

        if bnb_cli_output is not None:
            print(f"   Output from 'python -m bitsandbytes':\n---\n{bnb_cli_output}\n---")
        
        if bnb_cli_error_str:
            print(f"   [FAIL] 'python -m bitsandbytes' indicated an issue: {bnb_cli_error_str}")
            all_checks_passed = False
            failed_checks_summary.append(f"bitsandbytes CLI check: {bnb_cli_error_str}")
        elif bnb_cli_output and ("ERROR" in bnb_cli_output.upper() or "CUDA SETUP FAILED" in bnb_cli_output.upper() or "CUDA_SETUP_FAILED" in bnb_cli_output.upper()):
            print("   [FAIL] 'python -m bitsandbytes' output suggests critical issues (e.g., CUDA setup failed).")
            all_checks_passed = False
            failed_checks_summary.append("bitsandbytes CLI: Output indicates CUDA setup failure or errors.")
        elif bnb_cli_output:
             print("   [PASS] 'python -m bitsandbytes' command executed. Review output for details (e.g., CUDA version used).")
        else:
            print("   [WARN] Could not get interpretable status from 'python -m bitsandbytes'.")

    except Exception as e:
        print(f"[FAIL] Error during bitsandbytes Python checks: {e}")
        all_checks_passed = False
        failed_checks_summary.append(f"bitsandbytes Python checks: {str(e)}")
else:
    print(f"[FAIL] {bnb_error}")
    all_checks_passed = False
    failed_checks_summary.append(bnb_error)
print("-" * 30 + "\n")

# --- 3. xformers Check ---
print("--- 3. xformers Check ---")
xformers_ok, xformers_version, xformers_error = check_library_import_and_version("xformers")
if xformers_ok:
    print(f"[PASS] xformers imported successfully. Version: {xformers_version}")
    try:
        print("\n   Running 'python -m xformers.info' for detailed status:")
        xf_cli_output, xf_cli_error_str = run_check([sys.executable, "-m", "xformers.info"])

        if xf_cli_output is not None:
            print(f"   Output from 'python -m xformers.info':\n---\n{xf_cli_output}\n---")

        if xf_cli_error_str:
            print(f"   [FAIL] 'python -m xformers.info' indicated an issue: {xf_cli_error_str}")
            all_checks_passed = False
            failed_checks_summary.append(f"xformers.info check: {xf_cli_error_str}")
        elif xf_cli_output and ("WARNING" in xf_cli_output.upper() or "CUDA EXTENSION IS NOT AVAILABLE" in xf_cli_output.upper() or "NOT BUILT WITH CUDA SUPPORT" in xf_cli_output.upper()):
            print("   [FAIL] 'python -m xformers.info' output suggests missing CUDA extensions or other critical warnings.")
            all_checks_passed = False 
            failed_checks_summary.append("xformers.info: Output indicates critical warnings or missing CUDA support.")
        elif xf_cli_output and "is_triton_available: False" in xf_cli_output:
            print("   [WARN] 'xformers.info' reports 'is_triton_available: False'. This might impact Unsloth performance if Triton kernels are not utilized.")
            # Not a hard fail, but a significant warning for Unsloth.
        elif xf_cli_output:
            print("   [PASS] 'python -m xformers.info' command executed. Review output for details.")
        else:
             print("   [WARN] Could not get interpretable status from 'python -m xformers.info'.")

    except Exception as e:
        print(f"[FAIL] Error during xformers Python checks: {e}")
        all_checks_passed = False
        failed_checks_summary.append(f"xformers Python checks: {str(e)}")
else:
    print(f"[FAIL] {xformers_error}")
    all_checks_passed = False
    failed_checks_summary.append(xformers_error)
print("-" * 30 + "\n")

# --- 4. Unsloth Check ---
print("--- 4. Unsloth Check ---")
unsloth_pkg_ok, unsloth_pkg_version, unsloth_pkg_error = check_library_import_and_version("unsloth")
if unsloth_pkg_ok:
    print(f"[PASS] Unsloth package imported successfully. Version: {unsloth_pkg_version}")
    try:
        from unsloth import FastLanguageModel
        print("   [PASS] Unsloth's FastLanguageModel class imported successfully.")
    except ImportError:
        print("   [FAIL] Failed to import FastLanguageModel from Unsloth (though unsloth package itself imported).")
        all_checks_passed = False
        failed_checks_summary.append("Unsloth: Failed to import FastLanguageModel.")
    except Exception as e:
        print(f"   [FAIL] Error during Unsloth FastLanguageModel import: {e}")
        all_checks_passed = False
        failed_checks_summary.append(f"Unsloth FastLanguageModel import: {str(e)}")
else:
    print(f"[FAIL] {unsloth_pkg_error}")
    all_checks_passed = False
    failed_checks_summary.append(unsloth_pkg_error)
print("-" * 30 + "\n")

# --- 5. Other Core Libraries ---
print("--- 5. Other Core Libraries Check ---")
libs_to_check = ["transformers", "peft", "trl", "accelerate", "datasets", "evaluate"]
all_other_libs_ok_flag = True
for lib_name in libs_to_check:
    lib_ok, lib_version, lib_error = check_library_import_and_version(lib_name)
    if lib_ok:
        print(f"  [PASS] {lib_name} imported successfully. Version: {lib_version}")
    else:
        print(f"  [FAIL] {lib_error}")
        all_checks_passed = False # If any optional lib fails import, overall success might be impacted for the full guide
        all_other_libs_ok_flag = False
        failed_checks_summary.append(lib_error)

if all_other_libs_ok_flag:
    print("\n[PASS] All other core LLM libraries checked successfully.")
else:
    print("\n[FAIL] One or more other core LLM libraries failed to import.")
print("-" * 30 + "\n")

# --- Final Summary ---
print("--- Overall Validation Summary ---")
if all_checks_passed:
    print("[SUCCESS] All critical validation checks seem to have passed based on imports and basic CLI interactions!")
    print("           Please carefully review the detailed output above for any subtle warnings, especially from 'bitsandbytes' or 'xformers.info'.")
else:
    print("[FAILURE] One or more critical validation checks failed. Please review the output above.")
    print("\nSummary of Failed/Problematic Checks:")
    if not failed_checks_summary:
        print("  (No specific failure messages captured in summary, but all_checks_passed is False. Review logs.)")
    for item in failed_checks_summary:
        print(f"  - {item}")
print("--------------------------------")
