#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import pandas  # for reading csv file
from IPython.display import display, clear_output  # for animations later in this notebook

def linear_model(x, w0, w1):
    return w0 + x * w1


def rmse(X, T, w0, w1):
    Y = linear_model(X, w0, w1)
    return np.sqrt(np.mean((T - Y)**2))


# Need a function for this.  Let's optimize w_bias then w
def coordinate_descent(errorF, X, T, w0, w1, dw, nSteps):
    step = 0
    current_error = errorF(X, T, w0, w1)
    error_sequence = [current_error]
    W_sequence = [[w0, w1]]
    changed = False

    while step < nSteps:

        step += 1
        
        if not changed:
            dw = dw * 0.1
            
        changed = False
        
        # first vary w_bias
        up_error = errorF(X, T, w0 + dw, w1)
        down_error = errorF(X, T, w0 - dw, w1)
        
        if down_error < current_error:
            dw = -dw
            
        while True:
            new_w0 = w0 + dw
            new_error = errorF(X, T, new_w0, w1)
            if new_error >= current_error or step > nSteps:
                break
            changed = True
            w0 = new_w0
            W_sequence.append([w0, w1])
            error_sequence.append(new_error)
            current_error = new_error
            step += 1

        # now vary w
        up_error = errorF(X, T, w0, w1 + dw)
        down_error = errorF(X, T, w0, w1 - dw)
        
        if down_error < current_error:
            dw = -dw
            
        while True:
            new_w1 = w1 + dw
            new_error = errorF(X, T, w0, new_w1)
            if new_error >= current_error or step > nSteps:
                break
            changed = True
            w1 = new_w1
            W_sequence.append([w0, w1])
            error_sequence.append(new_error)
            current_error = new_error
            step += 1

    return w0, w1, error_sequence, W_sequence

def plot_sequence(error_sequence, W_sequence, label):
    plt.subplot(1, 2, 1)
    plt.plot(error_sequence, 'o-', label=label)
    plt.xlabel('Steps')
    plt.ylabel('Error')
    plt.legend()
    plt.subplot(1, 2, 2)
    W_sequence = np.array(W_sequence)
    plt.plot(W_sequence[:, 0], W_sequence[:, 1], '.-', label=label)
    plot_error_surface()

def plot_error_surface():
    n = 20
    w0s = np.linspace(-5, 5, n) 
    w1s = np.linspace(-0.5, 1.0, n) 
    w0s, w1s = np.meshgrid(w0s, w1s)
    surface = []
    for w0i in range(n):
        for w1i in range(n):
            surface.append(rmse(X, T, w0s[w0i, w1i], w1s[w0i, w1i]))
    plt.contourf(w0s, w1s, np.array(surface).reshape((n, n)), cmap='bone')
    # plt.colorbar()
    plt.xlabel('w_bias')
    plt.ylabel('w')
    
def show_animation(model, error_sequence, W_sequence, X, T, label):
    W_sequence = np.array(W_sequence)
    fig = plt.figure(figsize=(15, 8))
    plt.subplot(1, 3, 1)
    error_line, = plt.plot([], [])
    plt.xlim(0, len(error_sequence))
    plt.ylim(0, max(error_sequence))

    plt.subplot(1, 3, 2)
    plot_error_surface()
 
    w_line, = plt.plot([], [], '.-', label=label)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(X, T, 'o')
    model_line, = plt.plot([], [], 'r-', lw=3, alpha=0.5, label=label)
    plt.xlim(0, 24)
    plt.ylim(np.min(T), np.max(T))

    for i in range(len(W_sequence)):
        
        error_line.set_data(range(i), error_sequence[:i])
        w_line.set_data(W_sequence[:i, 0], W_sequence[:i, 1])
        Y = model(X, W_sequence[i, 0], W_sequence[i, 1])
        model_line.set_data(X, Y)

        plt.pause(0.001)

        clear_output(wait=True)
        display(fig)










data = pandas.read_csv('AirQualityUCI.csv', delimiter=';', decimal=',', usecols=range(15), na_values=-200)
data = data.dropna(axis=0)
data.shape
data.head(10)
data['Time'][:10]
[t for t in data['Time'][:10]]
[t[:2] for t in data['Time'][:10]]
hour = [int(t[:2]) for t in data['Time']]
data.columns
CO = data['CO(GT)']
CO[:10]
T = CO
T = np.array(T).reshape((-1, 1))  # make T have one column and as many rows as needed to hold the values of T
Tnames = ['CO']
X = np.array(hour).reshape((-1, 1))
Xnames = ['Hour']
print('X.shape =', X.shape, 'Xnames =', Xnames, 'T.shape =', T.shape, 'Tnames =', Tnames)
# or, using the latest formatting ability in python strings,
print(f'{X.shape=} {Xnames=} {T.shape=} {Tnames=}')
plt.plot(X, T, '.')
plt.xlabel(Xnames[0])
plt.ylabel(Tnames[0]);  # semi-colon here prevents printing the cryptic result of call to plt.ylabel()



w0 = 0
w1 = 1

Y = linear_model(X, w0, w1)

plt.plot(X, T, '.', label='Actual CO')
plt.plot(X, Y, 'r.-', label='Predicted CO')
plt.xlabel(Xnames[0])
plt.ylabel(Tnames[0])
plt.legend();  # make legend using the label strings

w1 = 0.1
Y = linear_model(X, w0, w1)

plt.plot(X, T, '.', label='Actual CO')
plt.plot(X, Y, 'r.-', label='Predicted CO')
plt.xlabel(Xnames[0])
plt.ylabel(Tnames[0])
plt.legend(); 

w1 = 0.3
Y = linear_model(X, w0, w1)

plt.plot(X, T, '.', label='Actual CO')
plt.plot(X, Y, 'r.-', label='Predicted CO')
plt.xlabel(Xnames[0])
plt.ylabel(Tnames[0])
plt.legend(); 

plt.show() 


w0 = 0.4
w1 = 0.5
dw = 0.1   # How much to change a weight's value.

current_error = rmse(X, T, w0, w1)
up_error = rmse(X, T, w0 + dw, w1)
down_error = rmse(X, T, w0 - dw, w1)

if down_error < current_error:
    dw = -dw
    new_error = down_error
else:
    new_error = up_error
    
while new_error <= current_error:
    current_error = new_error
    w0 = w0 + dw
    new_error = rmse(X, T, w0, w1)
    print(f'{w0=:5.2f} {new_error=:.5f}')


dw = 0.1
current_error = rmse(X, T, w0, w1)
up_error = rmse(X, T, w0, w1 + dw)
down_error = rmse(X, T, w0, w1 - dw)

if down_error < current_error:
    dw = -dw
    new_error = down_error
else:
    new_error = up_error
    
while new_error <= current_error:
    current_error = new_error
    w1 = w1 + dw
    new_error = rmse(X, T, w0, w1)
    print('w1 = {:.2f} new_error = {:.5f}'.format(w1, new_error))

w0 = -2
w1 = 0.5
nSteps = 200
dw = 10
w0, w1, error_sequence, W_sequence = coordinate_descent(rmse, X, T, w0, w1, dw, nSteps)
print(f'Coordinate Descent: Error is {rmse(X, T, w0, w1):.2f}   W is {w0:.2f}, {w1:.2f}')


show_animation(linear_model, error_sequence, W_sequence, X, T, 'coord desc')





def linear_model(X, W):
    # W is column vector
    return X @ W[1:, :] + W[0, :]

def rmse(model, X, T, W):
    Y = model(X, W)
    return np.sqrt(np.mean((T - Y)**2))


######################################################################

def vector_length(v):
    return np.sqrt(v.T @ v)

def run_and_twiddle(model_f, rmse_f, X, T, W, dW, nSteps):
    step = 0
    current_error = rmse_f(model_f, X, T, W)
    error_sequence = [current_error]
    W_sequence = [W.flatten()]
    nFails = 0
    
    while step < nSteps:
        # print(step)
        new_direction = np.random.uniform(-1, 1, size=(2, 1))
        # print(nFails, new_direction)
        new_direction = dW * new_direction / vector_length(new_direction)
        if nFails > 10:
            dW = dW * 0.8
        while step < nSteps:
            new_W = W.copy() + new_direction
            new_error = rmse_f(model_f, X, T, new_W)
            step += 1
            if new_error >= current_error:
                nFails += 1
                break
            nFails = 0
            # print('good', new_direction)
            W = new_W
            W_sequence.append(W.flatten())
            error_sequence.append(new_error)
            current_error = new_error

    return W, error_sequence, W_sequence


def plot_error_surface(model):
    n = 20
    wbiass = np.linspace(-5, 5, n)
    ws = np.linspace(-0.5, 1.0, n)
    wbiass, ws = np.meshgrid(wbiass, ws)
    surface = []
    for wbi in range(n):
        for wi in range(n):
            W = np.array([wbiass[wbi, wi], ws[wbi, wi]]).reshape(-1, 1)
            surface.append(rmse(model, X, T, W))
    plt.contourf(wbiass, ws, np.array(surface).reshape((n, n)), cmap='bone')
    # plt.colorbar()
    plt.xlabel('w_bias')
    plt.ylabel('w')
    
def show_animation(model, error_sequence, W_sequence, X, T, label):
    W_sequence = np.array(W_sequence)
    fig = plt.figure(figsize=(15, 8))
    plt.subplot(1, 3, 1)
    error_line, = plt.plot([], [])
    plt.xlim(0, len(error_sequence))
    plt.ylim(0, max(error_sequence))

    plt.subplot(1, 3, 2)
    plot_error_surface(model)
 
    w_line, = plt.plot([], [], '.-', label=label)
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(X, T, 'o')
    model_line, = plt.plot([], [], 'r-', lw=3, alpha=0.5, label=label)
    plt.xlim(0, 24)
    plt.ylim(np.min(T), np.max(T))

    for i in range(len(W_sequence)):
        
        error_line.set_data(range(i), error_sequence[:i])
        w_line.set_data(W_sequence[:i, 0], W_sequence[:i, 1])
        Y = model(X, W_sequence[i:i + 1, :].T)
        model_line.set_data(X, Y)

        plt.pause(0.001)

        clear_output(wait=True)
        display(fig)


w_bias = -2 # 10
w = 0.5
W = np.array([w_bias, w]).reshape(-1, 1)

nSteps = 400
dW = 10

W, error_sequence, W_sequence = run_and_twiddle(linear_model, rmse, X, T, W, dW, nSteps)
print('Run and Twiddle:  Error is {:.2f}   W is {:.2f}, {:.2f}'.format(rmse(linear_model, X, T, W), W[0,0], W[1,0]))

show_animation(linear_model, error_sequence, W_sequence, X, T, 'run & twiddle')


# Still using linear_model as defined above

#def linear_model(X, W):
#    # W is column vector
#    return X @ W[1:, :] + W[0,:]


def dYdW(X, T, W):
    # One row per sample in X,T.  One column per W component.
    # For first one, is constant 1.
    # For second one, is value of X
    return np.insert(X, 0, 1, axis=1)

def dEdY(X, T, W):
    Y = linear_model(X, W)
    return -2 * (T - Y)
    
def dEdW(X, T, W):
    result = dEdY(X, T, W).T @ dYdW(X, T, W) / (X.shape[0])
    return result.T

def gradient_descent(model_f, gradient_f, rmse_f, X, T, W, rho, nSteps):
    error_sequence = []
    W_sequence = []
    for step in range(nSteps):
        
        error_sequence.append(rmse_f(model_f, X, T, W))
        W_sequence.append(W.flatten())
        
        W -= rho * gradient_f(X, T, W)  # HERE IS THE WHOLE ALGORITHM!!
        
    return W, error_sequence, W_sequence

w_bias = -2 # 10
w = 0.5
W = np.array([w_bias, w]).reshape(-1, 1)

nSteps = 200
rho = 0.005

W, error_sequence, W_sequence = gradient_descent(linear_model, dEdW, rmse, X, T, W, rho, nSteps)
print('Gradient Descent:  Error is {:.2f}   W is {}'.format(rmse(linear_model, X, T, W), W))


show_animation(linear_model, error_sequence, W_sequence, X, T, 'gradient descent')

def gradient_descent_adam(model_f, gradient_f, rmse_f, X, T, W, rho, nSteps):
    # Commonly used parameter values
    alpha = rho
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    m = 0
    v = 0
    
    error_sequence = []
    W_sequence = []
    for step in range(nSteps):
        error_sequence.append(rmse_f(model_f, X, T, W))
        W_sequence.append(W.flatten())
        
        g = gradient_f(X, T, W)
        
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g * g
        mhat = m / (1 - beta1 ** (step+1))
        vhat = v / (1 - beta2 ** (step+1))
        
        W -= alpha * mhat / (np.sqrt(vhat) + epsilon)
        
    return W, error_sequence, W_sequence

w_bias = -2 # 10
w = 0.5
W = np.array([w_bias, w]).reshape(-1, 1)

rho = 0.01
nSteps = 200

W, error_sequence, W_sequence = gradient_descent_adam(linear_model, dEdW, rmse, X, T, W, rho, nSteps)
print('Adam:  Error is {:.2f}   W is {:.2f}, {:.2f}'.format(rmse(linear_model, X, T, W), W[0,0], W[1,0]))


def nonlinear_model(X_powers, W):
    # W is column vector
    return X_powers @ W

def dYdW(X_powers, T, W):
    return X_powers

def dEdY(X_powers, T, W):
    Y = nonlinear_model(X_powers, W)
    return -2 * (T - Y)
    
# dEdW from before does not need to be changed.

max_degree = 5

w_bias = -2 # 10
w_linear = 0.5

w_bias = 0
w_linear = 0

ws_nonlinear = np.zeros(max_degree + 1 - 2)

W = np.hstack((w_bias, w_linear, *ws_nonlinear)).reshape(-1, 1)
print(f'{W=}')

rho = 0.01
nSteps = 400
X_powers = X ** range(max_degree + 1)

W, error_sequence, W_sequence = gradient_descent_adam(nonlinear_model, dEdW, rmse, X_powers, T, W, rho, nSteps)
print('Adam:  Error is {:.2f}   W is {:.2f}, {:.2f} {:.2f}'.format(rmse(nonlinear_model, X, T, W), W[0,0], W[1,0], W[2,0]))

plt.figure(figsize=(10,8))
plt.plot(X + np.random.uniform(-0.1, 0.1, X.shape), T, '.', label='Training Data')

plt.plot(X, nonlinear_model(X_powers, W), 'ro', label='Prediction on Training Data')

plt.xlabel(Xnames[0])
plt.ylabel(Tnames[0])

plt.legend();


plt.figure(figsize=(10,8))
plt.plot(X + np.random.uniform(-0.1, 0.1, X.shape), T, '.', label='Training Data')

plt.plot(X, nonlinear_model(X_powers, W), 'r', label='Prediction on Training Data')
#   only change is here --------------------^
plt.xlabel(Xnames[0])
plt.ylabel(Tnames[0])

plt.legend();


plt.figure(figsize=(10,8))
plt.plot(X, T, '.', label='Training Data')

order = np.argsort(X, axis=0).ravel()  # change to 1-dimensional vector
plt.plot(X[order], nonlinear_model(X_powers, W)[order], 'r', label='Prediction on Training Data')

plt.xlabel(Xnames[0])
plt.ylabel(Tnames[0])

plt.legend();























