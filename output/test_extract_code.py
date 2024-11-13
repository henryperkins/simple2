import ast
from extract.code import extract_classes_and_functions_from_ast
from core.logger import LoggerSetup
logger = LoggerSetup.get_logger('extract.code', console_logging=True)
sample_code = '\nclass SampleClass:\n    def method_one(self):\n        pass\n\ndef standalone_function():\n    pass\n'
tree = ast.parse(sample_code)
result = extract_classes_and_functions_from_ast(tree, sample_code)
print('Classes extracted:')
for cls in result['classes']:
    print(f'Class name: {cls.get('name', 'No name')}')
print('\nFunctions extracted:')
for func in result['functions']:
    print(f'Function name: {func.get('name', 'No name')}')