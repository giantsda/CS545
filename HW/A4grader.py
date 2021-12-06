run_my_solution = False
assignmentNumber = '4'

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
    
required_funcs = ['train_this_partition', 'run_these_parameters']

for func in required_funcs:
    if func not in dir() or not callable(globals()[func]):
        print('CRITICAL ERROR: Function named \'{}\' is not defined'.format(func))
        print('  Check the spelling and capitalization of the function name.')

exec_grade = 0


######################################################################
print('''
===========================================================================================
Testing:

    def make_images(n_each_class):
        images = np.zeros((n_each_class * 2, 20, 20))  # nSamples, rows, columns
        radii = 3 + np.random.randint(10 - 5, size=(n_each_class * 2, 1))
        centers = np.zeros((n_each_class * 2, 2))
        for i in range(n_each_class * 2):
            r = radii[i, 0]
            centers[i, :] = r + 1 + np.random.randint(18 - 2 * r, size=(1, 2))
            x = int(centers[i, 0])
            y = int(centers[i, 1])
            if i < n_each_class:
                # squares
                images[i, x - r:x + r, y + r] = 1.0
                images[i, x - r:x + r, y - r] = 1.0
                images[i, x - r, y - r:y + r] = 1.0
                images[i, x + r, y - r:y + r + 1] = 1.0
            else:
                # diamonds
                images[i, range(x - r, x), range(y, y + r)] = 1.0
                images[i, range(x - r, x), range(y, y - r, -1)] = 1.0
                images[i, range(x, x + r + 1), range(y + r, y - 1, -1)] = 1.0
                images[i, range(x, x + r), range(y - r, y)] = 1.0
        T = np.array(['square'] * n_each_class + ['diamond'] * n_each_class).reshape(-1, 1)
        n, r, c = images.shape
        images = images.reshape(n, r, c, 1)  # add channel dimsension
        return images, T

    np.random.seed(4200)

    n_each_class = 10
    Xtrain, Ttrain = make_images(n_each_class * 2)
    Xval, Tval = make_images(n_each_class)
    Xtest, Ttest = make_images(n_each_class)
    print(Xtrain.shape, Ttrain.shape, Xval.shape, Tval.shape, Xtest.shape, Ttest.shape)

    struct = [ [[2, 5, 1]], [5] ]
    n_epochs = 20
    method= 'adam'
    learning_rate = 0.01
    batch_size = 5

    result = train_this_partition(Xtrain, Ttrain, Xval, Tval, Xtest, Ttest,
                                   struct, n_epochs, method, learning_rate, batch_size)
''')

try:

    pts = 20

    def make_images(n_each_class):
        '''Make 20x20 black and white images with diamonds or squares for the two classes, as line drawings.'''
        images = np.zeros((n_each_class * 2, 20, 20))  # nSamples, rows, columns
        radii = 3 + np.random.randint(10 - 5, size=(n_each_class * 2, 1))
        centers = np.zeros((n_each_class * 2, 2))
        for i in range(n_each_class * 2):
            r = radii[i, 0]
            centers[i, :] = r + 1 + np.random.randint(18 - 2 * r, size=(1, 2))
            x = int(centers[i, 0])
            y = int(centers[i, 1])
            if i < n_each_class:
                # squares
                images[i, x - r:x + r, y + r] = 1.0
                images[i, x - r:x + r, y - r] = 1.0
                images[i, x - r, y - r:y + r] = 1.0
                images[i, x + r, y - r:y + r + 1] = 1.0
            else:
                # diamonds
                images[i, range(x - r, x), range(y, y + r)] = 1.0
                images[i, range(x - r, x), range(y, y - r, -1)] = 1.0
                images[i, range(x, x + r + 1), range(y + r, y - 1, -1)] = 1.0
                images[i, range(x, x + r), range(y - r, y)] = 1.0
        T = np.array(['square'] * n_each_class + ['diamond'] * n_each_class).reshape(-1, 1)
        n, r, c = images.shape
        images = images.reshape(n, r, c, 1)  # add channel dimsension
        return images, T

    np.random.seed(4200)

    n_each_class = 10
    Xtrain, Ttrain = make_images(n_each_class * 2)
    Xval, Tval = make_images(n_each_class)
    Xtest, Ttest = make_images(n_each_class)
    print(Xtrain.shape, Ttrain.shape, Xval.shape, Tval.shape, Xtest.shape, Ttest.shape)

    struct = [ [[2, 5, 1]], [5] ]
    n_epochs = 20
    method= 'adam'
    learning_rate = 0.01
    batch_size = 5

    result = train_this_partition(Xtrain, Ttrain, Xval, Tval, Xtest, Ttest,
                                   struct, n_epochs, method, learning_rate, batch_size)

    correct = [(((2, 5, 1),), (5,)), 'adam', 20, 0.01, 5, 97.5, 75.0, 70.0]
    if result[:4] == correct[:4] and np.allclose(result[5:], correct[5:], atol=15):
        exec_grade += pts
        print('\n--- ', pts, '/', pts, 'points. train_this_partition correctly returns\n', result)
    else:
        print('\n---  0 /', pts, 'points. train_this_partition returns\n', result, 'but it should return\n', correct)
        

except Exception as ex:
    print('\n--- 0/', pts, 'points. train_this_partition raised the exception:\n')
    print(ex)


######################################################################
print('''
===========================================================================================
Testing

    np.random.seed(4200)

    n_each_class = 10
    Xtrain, Ttrain = make_images(n_each_class * 2)
    Xval, Tval = make_images(n_each_class)
    Xtest, Ttest = make_images(n_each_class)
    print(Xtrain.shape, Ttrain.shape, Xval.shape, Tval.shape, Xtest.shape, Ttest.shape)

    structs = [ [[], []],
                [[], [10]],
                [[[4, 5, 1], [5, 3, 1]], [5]]
               ]
    n_epochs = [10, 20]
    methods= ['adam']
    learning_rates = [0.01, 0.1]
    batch_sizes = [5]

    results = []
    for struct in structs:
        for epochs in n_epochs:
            for method in methods:
                for learning_rate in learning_rates:
                    for batch_size in batch_sizes:
                    
                        # This next for loop simulates how you will use generate_partitions
                        # in run_these_parameters
                        for Xtrain, Ttrain, Xval, Tval, Xtest, Ttest in [[Xtrain, Ttrain, Xval, Tval, Xtest, Ttest]]:
                            result = train_this_partition(Xtrain, Ttrain, Xval, Tval, Xtest, Ttest,
                                                          struct, epochs, method, learning_rate, batch_size)
                            results.append(result)
''')

try:

    pts = 20

    np.random.seed(4200)

    n_each_class = 10
    Xtrain, Ttrain = make_images(n_each_class * 2)
    Xval, Tval = make_images(n_each_class)
    Xtest, Ttest = make_images(n_each_class)
    print(Xtrain.shape, Ttrain.shape, Xval.shape, Tval.shape, Xtest.shape, Ttest.shape)

    structs = [ [[], []],
                [[], [10]],
                [[[4, 5, 1], [5, 3, 1]], [5]]
               ]
    n_epochs = [10, 20]
    methods= ['adam']
    learning_rates = [0.01, 0.1]
    batch_sizes = [5]

    results = []
    for struct in structs:
        for epochs in n_epochs:
            for method in methods:
                for learning_rate in learning_rates:
                    for batch_size in batch_sizes:
                    
                        # This next for loop simulates how you will use generate_partitions
                        # in run_these_parameters
                        for Xtrain, Ttrain, Xval, Tval, Xtest, Ttest in [[Xtrain, Ttrain, Xval, Tval, Xtest, Ttest]]:
                            result = train_this_partition(Xtrain, Ttrain, Xval, Tval, Xtest, Ttest,
                                                          struct, epochs, method, learning_rate, batch_size)
                            results.append(result)

    correct = [[((), ()), 'adam', 10, 0.01, 5, 100.0, 65.0, 65.0],
               [((), ()), 'adam', 10, 0.1, 5, 100.0, 75.0, 70.0],
               [((), ()), 'adam', 20, 0.01, 5, 100.0, 70.0, 65.0],
               [((), ()), 'adam', 20, 0.1, 5, 100.0, 70.0, 80.0],
               [((), (10,)), 'adam', 10, 0.01, 5, 100.0, 70.0, 80.0],
               [((), (10,)), 'adam', 10, 0.1, 5, 100.0, 60.0, 85.0],
               [((), (10,)), 'adam', 20, 0.01, 5, 100.0, 70.0, 75.0],
               [((), (10,)), 'adam', 20, 0.1, 5, 100.0, 70.0, 85.0],
               [(((4, 5, 1), (5, 3, 1)), (5,)), 'adam', 10, 0.01, 5, 97.5, 65.0, 75.0],
               [(((4, 5, 1), (5, 3, 1)), (5,)), 'adam', 10, 0.1, 5, 87.5, 65.0, 70.0],
               [(((4, 5, 1), (5, 3, 1)), (5,)), 'adam', 20, 0.01, 5, 100.0, 70.0, 70.0],
               [(((4, 5, 1), (5, 3, 1)), (5,)), 'adam', 20, 0.1, 5, 100.0, 70.0, 70.0]]

    results_first_5 = [res[:5] for res in results]
    results_means = np.mean([res[5:] for res in results], axis=0)
    correct_first_5 = [cor[:5] for cor in correct]
    correct_means = np.mean([cor[5:] for cor in correct], axis=0)

    first_5 = True
    for res in results_first_5:
        if res not in correct_first_5:
            first_5 = False
            break

    if first_5 and np.allclose(results_means, correct_means, atol=10):
        exec_grade += pts
        print('\n---', pts, '/', pts, 'points. train_this_partition correctly returns')
        for res in results:
            print(res)
    else:
        print('\n---  0 /', pts, 'points. train_this_partition returns')
        for res in results:
            print(res)
        print('but it should return')
        for res in correct:
            print(res)
        

except Exception as ex:
    print('\n--- 0 / ', pts, 'points. train_this_partition raised the exception:\n')
    print(ex)

    




######################################################################

print('''
===========================================================================================
Testing

    np.random.seed(4200)

    n_each_class = 40
    X, T = make_images(n_each_class * 2)

    structs = [ [[], []],
                [[], [10]],
                [[[4, 5, 1], [5, 3, 1]], [5]]
               ]
    methods= ['adam']
    n_epochs = [10]
    learning_rates = [0.01, 0.1]
    batch_sizes = [-1, 5]  # -1 means train on all training samples, not multiple batches
    n_folds = 3
    
    print('This may take several minutes...')
    
    df = run_these_parameters(X, T, n_folds,
                              structs, methods, n_epochs, learning_rates, batch_sizes)

    # Here is a function you can use to print all columns and rows of a DataFrame
    def print_df(df):
        with pandas.option_context('display.width', 100,
                                   'display.max_rows', None,
                                   'display.max_columns', None):
            print(df)
''')


try:

    np.random.seed(4200)

    n_each_class = 40
    X, T = make_images(n_each_class * 2)

    structs = [ [[], []],
                [[], [10]],
                [[[4, 5, 1], [5, 3, 1]], [5]]
               ]
    methods= ['adam']
    n_epochs = [10]
    learning_rates = [0.01, 0.1]
    batch_sizes = [-1, 5]  # -1 means train on all training samples, not multiple batches
    n_folds = 3
    
    print('This may take several minutes...')
    
    df = run_these_parameters(X, T, n_folds,
                              structs, methods, n_epochs, learning_rates, batch_sizes)

    pts = 10
    
    if df.shape == (72, 8):
        exec_grade += pts
        print('\n--', pts, '/', pts, 'points. DataFrame returns has correct shape, (72, 8)')
    else:
        print('\n-- 0 /', pts, 'points. DataFrame has shape', df.shape, ', but should be (72, 8)')


    pts = 20
    
    correct = np.array([94.09340659, 75.33768315, 75.6753663 ])
    
    means = df[['train %', 'val %', 'test %']].mean().values

    def print_df(df):
        with pandas.option_context('display.width', 100,
                                   'display.max_rows', None,
                                   'display.max_columns', None):
            print(df)


    if np.allclose(means, correct, atol=10):
        exec_grade += pts
        print('\n---', pts, '/', pts, 'points. Returned DataFrame has correct percent means of', means, ':')
    else:
        print('\n---  0 /', pts, 'points. Returned DataFrame has accuracy means of', means, ', but they should be', correct)
        

except Exception as ex:
    print('\n--- 0 / ', pts, 'points. run_these_parameters raised the exception:\n')
    print(ex)

    





name = os.getcwd().split('/')[-1]

print()
print('='*60)
print(name, 'Execution Grade is', exec_grade, '/ 70')
print('='*60)

print('''

__ / 5 points.  Discuss what you observe after each call to run_these_parameters with
         at least two sentences for each run.

__ / 5 points.  Discuss which parameter values seem to work the best according to
         the validation accuracy.

__ / 5 points.  Discuss how well the validation and test accuracies equal each other
          and what you might do to make them more equal.

__ / 5 points.  Show and discuss a confusion matrix on test data for a network trained
          with your best parameter values.

__ / 10 points.  For each method, try various hidden layer structures, learning rates,
          and numbers of epochs. Use the validation percent accuracy to pick the best
          hidden layers, learning rates and numbers of epochs for each method. Report
          training, validation and test accuracy for your best validation results
          for each of the three methods.
''')

print()
print('='*70)
print(name, 'Results and Discussion Grade is ___ / 30')
print('='*70)


print()
print('='*70)
print(name, 'FINAL GRADE is  _  / 100')
print('='*70)


print('''
Extra Credit: 
Repeat the above experiment using a convolutional neural network defined in Pytorch.
Implement this yourself by directly calling torch.nn functions.
''')

print('\n', name, 'EXTRA CREDIT is 0 / 1')


if run_my_solution:
    print('##############################################')
    print("RUNNING INSTRUCTOR's SOLUTION!!!!!!!!!!!!")
    print('##############################################')

    
