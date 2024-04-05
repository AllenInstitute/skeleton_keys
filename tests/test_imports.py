import os
import unittest
import importlib
import inspect
import skeleton_keys

def import_functions_from_module(module):
    functions = inspect.getmembers(module, inspect.isfunction)
    if functions:
        return functions[0][1]
    return None

class TestImportFunctions(unittest.TestCase):

    def test_import_all_modules(self):
        package_path = skeleton_keys.__path__[0]
        package_name = skeleton_keys.__name__
        
        for root, _, files in os.walk(package_path):
            for file_name in files:
                if file_name.endswith('.py'):
                    module_path = os.path.relpath(os.path.join(root, file_name), package_path)
                    module_name = module_path.replace(os.path.sep, '.')[:-3]
                    if module_name.endswith('__init__'):
                        continue
                    full_module_name = f"{package_name}.{module_name}"
                    print(full_module_name)
                    try:
                        module = importlib.import_module(full_module_name)
                        assert module is not None, f"Failed to import module: {full_module_name}"
                        first_function = import_functions_from_module(module)
                        assert first_function is not None, f"No functions found in module: {full_module_name}"
                    
                    except Exception as e:
                        assert False, f"Failed to import function from {full_module_name}: {e}"

if __name__ == '__main__':
    unittest.main()
