"""
Quick Structure Test - No Heavy Dependencies Required

This test verifies the code structure and imports without requiring
PyTorch or other heavy dependencies.
"""

import ast
import sys
import os


def test_file_imports(filepath, expected_classes=None):
    """Test if a file can be parsed and contains expected classes"""
    print(f"Testing {filepath}...")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse AST
        tree = ast.parse(content)

        # Find all class definitions
        classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

        print(f"  [OK] Syntax valid")
        print(f"  [OK] Found {len(classes)} classes: {', '.join(classes[:5])}")
        print(f"  [OK] Found {len(functions)} functions")

        # Check expected classes
        if expected_classes:
            for expected in expected_classes:
                if expected in classes:
                    print(f"  [OK] Found expected class: {expected}")
                else:
                    print(f"  [FAIL] Missing expected class: {expected}")
                    return False

        return True

    except Exception as e:
        print(f"  [FAIL] Error: {e}")
        return False


def main():
    print("="*70)
    print("Quick Structure Test (No PyTorch Required)")
    print("="*70)
    print()

    results = []

    # Test grpo_reward.py
    print("[1/3] Testing grpo_reward.py...")
    result = test_file_imports('grpo_reward.py', expected_classes=['GRPORewardModel'])
    results.append(('grpo_reward.py', result))
    print()

    # Test train_grpo.py
    print("[2/3] Testing train_grpo.py...")
    result = test_file_imports('train_grpo.py', expected_classes=['GRPOTrainer'])
    results.append(('train_grpo.py', result))
    print()

    # Test run_smoke_test.py
    print("[3/3] Testing run_smoke_test.py...")
    result = test_file_imports('run_smoke_test.py', expected_classes=['DummyDataset', 'DummyDataLoader'])
    results.append(('run_smoke_test.py', result))
    print()

    # Summary
    print("="*70)
    print("Summary")
    print("="*70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for filename, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status} {filename}")

    print()
    print(f"Result: {passed}/{total} files passed")

    if passed == total:
        print()
        print("[SUCCESS] All structure tests passed!")
        print()
        print("Code structure is correct. To run full tests:")
        print("1. Wait for PyTorch installation to complete")
        print("2. Run: python run_smoke_test.py")
        return True
    else:
        print()
        print("[FAIL] Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
