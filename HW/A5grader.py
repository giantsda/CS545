run_my_solution = False
assignmentNumber = '5'

import os
import copy
import signal
import os
import numpy as np
import subprocess
import pandas

if run_my_solution:

    from A4mysolution import *
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
        print('More than one ipynb file found:', nb_names, '. Using first one.')
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
# from neuralnetworks_A4 import NeuralNetwork, NeuralNetworkClassifier, NeuralNetworkClassifier_CNN
# print('Deleting nn.py')
# subprocess.call(['rm', '-f', 'nn.py'])
    
required_funcs = ['Marble_Variable_Goal', 'Qnet']

for func in required_funcs:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined'.format(func))
        print('  Check the spelling and capitalization of the function name.')

exec_grade = 0


######################################################################
print('''
===========================================================================================
Testing:

    marble = Marble_Variable_Goal((-2, -1, 0, 1, 2))
    s = marble.initial_state()

''')

try:

    pts = 20

    marble = Marble_Variable_Goal((-2, -1, 0, 1, 2))
    s = marble.initial_state()

    if len(s) == 3:
        exec_grade += pts
        print('\n--- ', pts, '/', pts, 'points. initial_state correctly returns a state with 3 components.')
    else:
        print('\n---  0 /', pts, 'points. initial_state returns state with', len(s), 'but, it should have 3 components.')
        

except Exception as ex:
    print('\n--- 0/', pts, 'points. initial_state raised the exception:\n')
    print(ex)


######################################################################
print('''
===========================================================================================
Testing

    marble = Marble_Variable_Goal((-2, -1, 0, 1, 2))
    s = marble.initial_state()
    s = marble.next_state(s, 1)
''')

try:

    pts = 20

    marble = Marble_Variable_Goal((-2, -1, 0, 1, 2))
    s = marble.initial_state()
    s = marble.next_state(s, 1)

    if len(s) == 3:
        exec_grade += pts
        print('\n--- ', pts, '/', pts, 'points. next_state correctly returns a state with 3 components.')
    else:
        print('\n---  0 /', pts, 'points. next_state returns state with', len(s), 'but, it should have 3 components.')
        

except Exception as ex:
    print('\n--- 0/', pts, 'points. next_state raised the exception:\n')
    print(ex)

    
######################################################################
print('''
===========================================================================================
Testing

    marble = Marble_Variable_Goal((-2, -1, 0, 1, 2))
    s = []
    for i in range(10):
        s.append(marble.initial_state()[2])


    if len(np.unique(s)) > 1:
        success  (20 points)
''')

try:

    pts = 20

    marble = Marble_Variable_Goal((-2, -1, 0, 1, 2))
    s = []
    for i in range(10):
        s.append(marble.initial_state()[2])

    n_unique = len(np.unique(s))
    if n_unique > 1:
        exec_grade += pts
        print('\n--- ', pts, '/', pts, 'points. initial_state correctly assigns the goal randomly..')
    else:
        print('\n---  0 /', pts, 'points. initial_state incorrectly does not change the goal.')
        

except Exception as ex:
    print('\n--- 0/', pts, 'points. initial_state raised the exception:\n')
    print(ex)

    


name = os.getcwd().split('/')[-1]

print()
print('='*60)
print(name, 'Execution Grade is', exec_grade, '/ 60')
print('='*60)

print('''

10 / 10 points.  Discuss what you modified in the code.

20 / 20 points. Code to test your trained agent by collecting final
          distances to goal for series of goals. Print resulting value
          of distances_to_goal and show a plot of these distances.

10 / 10 points.  Discuss results of your experimentation with parameter values.
          How sensitive are results to each parameter?

''')

print()
print('='*70)
print(name, 'Results and Discussion Grade is ___ / 40')
print('='*70)


print()
print('='*70)
print(name, 'FINAL GRADE is  _  / 100')
print('='*70)


print('''
Extra Credit: 

1 point: Modify your solution to this assignment by creating and
   using a Marble2D_Variable_Goal class that simulates the marble moving in 
   two-dimensions, on a plane. Some of the current plots will not
   work for this case. Just show the ones that are still appropriate.

1 point: Experiment with seven valid actions rather than three. How does
   this change the behavior of the controlled marble?

''')

print('\n', name, 'EXTRA CREDIT is 0 / 1')


if run_my_solution:
    print('##############################################')
    print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    print('##############################################')

    
