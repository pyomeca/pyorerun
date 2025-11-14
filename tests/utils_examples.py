"""
Utility functions for running example scripts in tests.
"""

import os
import sys
from pathlib import Path
import importlib.util


class ExampleRunner:
    @staticmethod
    def example_folder() -> Path:
        """Get the path to the examples folder."""
        return Path(__file__).parent / "../examples"

    @staticmethod
    def run_example_module(module_path: Path):
        """
        Dynamically import and run an example module.
        
        Args:
            module_path: Path to the example Python file to run
        """
        # Add the module's directory to sys.path temporarily
        module_dir = str(module_path.parent)
        original_cwd = os.getcwd()
        original_path = sys.path.copy()
        
        try:
            # Change to the example directory
            os.chdir(module_dir)
            sys.path.insert(0, module_dir)
            
            # Load the module
            spec = importlib.util.spec_from_file_location("test_module", module_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                
                # Execute the module
                spec.loader.exec_module(module)
                
                # If the module has a main function, call it
                if hasattr(module, "main"):
                    module.main()
                
        finally:
            # Restore original directory and path
            os.chdir(original_cwd)
            sys.path = original_path
