"""Public entry point for the retained, collision-as-failure evaluation."""
from tools.run_native_ablation100 import main
from evaluate_model_suite import _apply_evaluation_stack  # milestone API compatibility

if __name__ == '__main__':
    main()
