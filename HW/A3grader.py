run_my_solution = False
assignmentNumber = '3'

import os
import copy
import signal
import os
import numpy as np
import subprocess

if run_my_solution:
    from A3mysolution import *
    print('##############################################')
    print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    print('##############################################')

else:
    
    print('\n======================= Code Execution =======================\n')


    import subprocess, glob, pathlib
    n = assignmentNumber
    nb_names = glob.glob(f'*-A{n}-[0-9].ipynb') + glob.glob(f'*-A{n}.ipynb')
    nb_names = np.unique(nb_names)
    nb_names = sorted(nb_names, key=os.path.getmtime)
    if len(nb_names) > 1:
        print(f'More than one ipynb file found: {nb_names}. Using first one.')
    elif len(nb_names) == 0:
        raise Exception(f'No jupyter notebook file found with name ending in -A{n}.')
    filename = nb_names[0]
    print('Extracting python code from notebook named \'{}\' and storing in notebookcode.py'.format(filename))
    if not filename:
        raise Exception('Please rename your notebook file to <Your Last Name>-A{}.ipynb'.format(assignmentNumber))
    with open('notebookcode.py', 'w') as outputFile:
        subprocess.call(['jupyter', 'nbconvert', '--to', 'script',
                         filename, '--stdout'], stdout=outputFile)
    # from https://stackoverflow.com/questions/30133278/import-only-functions-from-a-python-file
    import sys
    import ast
    import types
    with open('notebookcode.py') as fp:
        tree = ast.parse(fp.read(), 'eval')
    print('Removing all statements that are not function or class defs or import statements.')
    for node in tree.body[:]:
        if (not isinstance(node, ast.FunctionDef) and
            not isinstance(node, ast.Import) and
            not isinstance(node, ast.ImportFrom) and
            not isinstance(node, ast.ClassDef)):
            tree.body.remove(node)
    # Now write remaining code to py file and import it
    module = types.ModuleType('notebookcodeStripped')
    code = compile(tree, 'notebookcodeStripped.py', 'exec')
    sys.modules['notebookcodeStripped'] = module
    exec(code, module.__dict__)
    # import notebookcodeStripped as useThisCode
    from notebookcodeStripped import *

# print('Copying your neuralnetworks.py to nn.py.')
# subprocess.call(['cp', 'neuralnetworks.py', 'nn.py'])
# print('Importing NeuralNetwork and NeuralNetworkClassifier from nn.py.')
from neuralnetworks import NeuralNetwork, NeuralNetworkClassifier
# print('Deleting nn.py')
# subprocess.call(['rm', '-f', 'nn.py'])
    
required_funcs = ['NeuralNetwork', 'NeuralNetworkClassifier']

for func in required_funcs:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined'.format(func))
        print('  Check the spelling and capitalization of the function name.')

exec_grade = 0


#### constructor ##################################################################

print('''
## Testing inheritance ####################################################################

    correct = issubclass(NeuralNetworkClassifier, NeuralNetwork)
''')
      
try:
    pts = 10
    correct = issubclass(NeuralNetworkClassifier, NeuralNetwork)

    if correct:
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. NeuralNetworkClassifier correctly extends NeuralNetwork.')
    else:
        print(f'\n---  0/{pts} points. NeuralNetworkClassifier should extend NeuralNetwork but it does not.')
except Exception as ex:
    print(f'\n--- 0/{pts} points. NeuralNetworkClassifier raised the exception:\n')
    print(ex)




print('''
## Testing inheritance ####################################################################

    import inspect
    forward_func = [f for f in inspect.classify_class_attrs(NeuralNetworkClassifier) if (f.name == 'forward' or f.name == '_forward')]
    correct = forward_func[0].defining_class == NeuralNetwork
''')
      
try:
    pts = 5

    import inspect
    forward_func = [f for f in inspect.classify_class_attrs(NeuralNetworkClassifier) if (f.name == 'forward' or f.name == '_forward')]
    correct = forward_func[0].defining_class == NeuralNetwork

    if correct:
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. NeuralNetworkClassifier forward function correctly inherited from NeuralNetwork.')
    else:
        print(f'\n---  0/{pts} points. NeuralNetworkClassifier forward function should be inherited from NeuralNetwork.')
except Exception as ex:
    print(f'\n--- 0/{pts} points. Test raised the exception:\n')
    print(ex)


print('''
## Testing inheritance ####################################################################

    import inspect
    str_func = [f for f in inspect.classify_class_attrs(NeuralNetworkClassifier) if (f.name == '__str__')]
    correct = str_func[0].defining_class == NeuralNetworkClassifier
''')
      
try:
    pts = 5

    import inspect
    str_func = [f for f in inspect.classify_class_attrs(NeuralNetworkClassifier) if (f.name == '__str__')]
    correct = str_func[0].defining_class == NeuralNetworkClassifier

    if correct:
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. NeuralNetworkClassifier __str__ function correctly defined in NeuralNetworkClassifier.')
    else:
        print(f'\n---  0/{pts} points. NeuralNetworkClassifier __str__ function should be defined in NeuralNetworkClassifier.')
except Exception as ex:
    print(f'\n--- 0/{pts} points. Test raised the exception:\n')
    print(ex)



print('''
## Testing constructor ####################################################################

    nnet = NeuralNetworkClassifier(2, [], 5)
    W_shapes = [W.shape for W in nnet.Ws]
''')
      
try:
    pts = 10
    nnet = NeuralNetworkClassifier(2, [], 5)
    W_shapes = [W.shape for W in nnet.Ws]
    correct = [(3, 5)]

    if correct == W_shapes:
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. W_shapes is correct value of {W_shapes}')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values {W_shapes}.')
        print(f'                 Correct values are {correct}')
except Exception as ex:
    print(f'\n--- 0/{pts} points. NeuralNetworkClassifier raised the exception\n')
    print(ex)
    

print('''
## Testing constructor ####################################################################

    G_shapes = [G.shape for G in nnet.Grads]
''')
      
try:
    pts = 10
    G_shapes = [G.shape for G in nnet.Grads]
    correct = [(3, 5)]

    if correct == G_shapes:
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. G_shapes is correct value of {G_shapes}')
    else:
        print(f'\n---  0/{pts} points. Returned incorrect values {G_shapes}.')
        print(f'                 Correct values are {correct}')
except Exception as ex:
    print(f'\n--- 0/{pts} points. Accessing nnet.Gs raised the exception\n')
    print(ex)



# def print_layers(what, lst):
#     print(f'{what}:')
#     for (i, element) in enumerate(lst):
#         print(f' Layer {i}:')
#         print(f' {element}')



#### train  ##################################################################
print('''
## Testing train ####################################################################

    np.random.seed(42)
    X = np.random.uniform(0, 1, size=(100, 2))
    T = (np.abs(X[:, 0:1] - 0.5) > 0.3).astype(int)
    nnet = NeuralNetworkClassifier(2, [10, 5], len(np.unique(T)))
    nnet.train(X, T, 20, method='scg')

    Then check  nnet.get_error_trace()
''')

try:
    pts = 10
    np.random.seed(42)
    X = np.random.uniform(0, 1, size=(100, 2))
    T = (np.abs(X[:, 0:1] - 0.5) > 0.3).astype(int)
    nnet = NeuralNetworkClassifier(2, [10, 5], len(np.unique(T)))
    nnet.train(X, T, 20, method='scg')

    last_error = nnet.get_error_trace()[-1]
    correct = 0.9297448356260026

    if np.allclose(last_error, correct, atol=0.1):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Correct values in error_trace')
    else:
        print(f'\n---  0/{pts} points. Incorrect values in error_trace')
        print(f'                 Your value is {last_error[0]}, but it should be {correct}')

except Exception as ex:
    print(f'\n--- 0/{pts} points. nnet.train or get_error_trace() raised the exception\n')
    print(ex)
    


#### train  ##################################################################
print('''
## Testing train ####################################################################

    np.random.seed(43)
    X = np.random.uniform(0, 1, size=(100, 2))
    T = (np.abs(X[:, 0:1] - X[:, 1:2]) < 0.5).astype(int)
    T[T == 0] = 10
    T[T == 1] = 20
    # Unique class labels are now 10 and 20!

    nnet = NeuralNetworkClassifier(2, [10, 5], 2)
    nnet.train(X, T, 200, method='scg')
    classes, prob = nnet.use(X)
''')

try:
    pts = 10
    np.random.seed(43)
    X = np.random.uniform(0, 1, size=(20, 2))
    T = (np.abs(X[:, 0:1] - X[:, 1:2]) < 0.5).astype(int)
    T[T == 0] = 10
    T[T == 1] = 20
    # Unique class labels are now 10 and 20!

    nnet = NeuralNetworkClassifier(2, [10, 5], 2)
    nnet.train(X, T, 200, method='scg')
    classes, prob = nnet.use(X)
    correct_classes = \
        np.array([[20],
                  [20],
                  [10],
                  [20],
                  [10],
                  [20],
                  [20],
                  [10],
                  [20],
                  [10],
                  [20],
                  [10],
                  [20],
                  [10],
                  [20],
                  [10],
                  [20],
                  [20],
                  [20],
                  [20]])
    correct_prob = \
        np.array([[7.87686254e-10, 9.99999999e-01],
                  [2.64073742e-10, 1.00000000e+00],
                  [1.00000000e+00, 2.17739214e-11],
                  [2.37507101e-10, 1.00000000e+00],
                  [1.00000000e+00, 5.72602779e-13],
                  [2.63951189e-10, 1.00000000e+00],
                  [3.07141256e-10, 1.00000000e+00],
                  [9.99999995e-01, 5.31601303e-09],
                  [5.18960837e-10, 9.99999999e-01],
                  [1.00000000e+00, 5.29910868e-15],
                  [2.31535786e-10, 1.00000000e+00],
                  [1.00000000e+00, 4.76259538e-17],
                  [2.31088925e-10, 1.00000000e+00],
                  [1.00000000e+00, 3.30767340e-16],
                  [3.09810289e-10, 1.00000000e+00],
                  [9.99999999e-01, 7.34252931e-10],
                  [2.31089312e-10, 1.00000000e+00],
                  [2.32724737e-10, 1.00000000e+00],
                  [2.69944404e-10, 1.00000000e+00],
                  [2.59802216e-10, 1.00000000e+00]])

    if np.allclose(classes, correct_classes, atol=0.1):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Correct values in classes')
    else:
        print(f'\n---  0/{pts} points. Incorrect values in classes')
        print(f'                 Your value is\n {classes}.')
        print(f'                 Correct value is\n {correct_classes}')

    if np.allclose(prob, correct_prob, atol=0.1):
        exec_grade += pts
        print(f'\n--- {pts}/{pts} points. Correct values in prob')
    else:
        print(f'\n---  0/{pts} points. Incorrect values in prob')
        print(f'                 Your value is\n {prob}.')
        print(f'                 Correct value is\n {correct_prob}')


except Exception as ex:
    print(f'\n--- 0/{pts} points. nnet.train or use raised the exception\n')
    print(ex)
    
    


    


name = os.getcwd().split('/')[-1]

print()
print('='*70)
print(f'{name} Execution Grade is {exec_grade} / 70')
print('='*70)

print('''

__ / 5 points. Correctly downloaded and read the MNIST data.

__ / 10 points. Experimented with different values of training parameters.

__ / 5 points. Show confusion matrix for best neural network.

__ / 10 points. Described results with at least 10 sentences.''')

print()
print('='*70)
print(f'{name} Results and Discussion Grade is ___ / 30')
print('='*70)


print()
print('='*70)
print(f'{name} FINAL GRADE is  _  / 100')
print('='*70)


print('''
Extra Credit: 
Repeat the above experiments with a different data set. Randonly partition 
your data into training, validaton and test parts if not already provided.
Write in markdown cells descriptions of the data and your results.''')

print(f'\n{name} EXTRA CREDIT is 0 / 1')


if run_my_solution:
    print('##############################################')
    print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    print('##############################################')

    
