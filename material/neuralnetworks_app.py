import streamlit as st

import numpy as np
import matplotlib.pyplot as plt

import neuralnetworks as nn

"""
# Neural Networks for Regression and Classification

Based on material from 
[CS545: Machine Learning](https://www.cs.colostate.edu/~anderson/cs545),
a graduate-level course at [Colorado State University](https://www.cs.colostate.edu).
"""

"""
## Regression
"""

st.sidebar.markdown("""
## Neural Network Parameters
""")

n_hidden_units_by_layer = [10, 10]
n_epochs = 600

n_units = st.sidebar.text_input("Numbers of units by layer (separated by spaces)", "10 10")
n_hidden_units_by_layer = [int(n) for n in n_units.split()]
# st.write(n_hidden_units_by_layer)

n_epochs = st.sidebar.number_input("Number of epochs", 100)

method = "scg"
method = st.sidebar.selectbox("Optimization mehod", ('scg', 'adam', 'sgd'))

learning_rate = 0.01
learning_rate = st.sidebar.number_input("Learning rate", learning_rate)

with st.echo():
    import neuralnetworks as nn

    Xtrain = np.linspace(-10, 10, 100).reshape(-1, 1)
    Ttrain = np.sin(Xtrain)

    n_inputs = Xtrain.shape[1]
    n_outputs = Ttrain.shape[1]

    nnet = nn.NeuralNetwork(n_inputs, n_hidden_units_by_layer, n_outputs)
    nnet.train(Xtrain, Ttrain, n_epochs=n_epochs, method=method,
               learning_rate=learning_rate)

    Xtest = np.linspace(-12, 12, 500).reshape(-1, 1)
    Y = nnet.use(Xtest)

st.write(nnet)
st.write(f'Trained with {method}')

fig = plt.figure()
plt.subplot(1, 2, 1)
plt.plot(nnet.get_error_trace())
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')

plt.subplot(1, 2, 2)
plt.plot(Xtrain, Ttrain, 'o', label='Training Data')
plt.plot(Xtest, Y, label='Test Data')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.tight_layout()

st.pyplot(fig)


"""
## Classification
"""

n = 500
x1 = np.linspace(5, 20, n) + np.random.uniform(-2, 2, n)
y1 = ((20-12.5)**2-(x1-12.5)**2) / (20-12.5)**2 * 10 + 14 + np.random.uniform(-2, 2, n)
x2 = np.linspace(10, 25, n) + np.random.uniform(-2, 2, n)
y2 = ((x2-17.5)**2) / (25-17.5)**2 * 10 + 5.5 + np.random.uniform(-2, 2, n)
angles = np.linspace(0, 2*np.pi, n)
x3 = np.cos(angles) * 15 + 15 + np.random.uniform(-2, 2, n)
y3 = np.sin(angles) * 15 + 15 + np.random.uniform(-2, 2, n)
X =  np.vstack((np.hstack((x1, x2, x3)),  np.hstack((y1, y2, y3)))).T
T = np.repeat(range(1, 4), n).reshape((-1, 1))

with st.echo():

    import neuralnetworks as nn

    nnet = nn.NeuralNetworkClassifier(2, n_hidden_units_by_layer, 3) # 3 classes, will actually make 2-unit output layer
    nnet.train(X, T, n_epochs=1000,  method='scg')

st.write(nnet)

colors = ['blue', 'red', 'green']

xs = np.linspace(0, 30, 40)
x, y = np.meshgrid(xs, xs)
Xtest = np.vstack((x.flat, y.flat)).T
Ytest = nnet.use(Xtest)
predTest, probs = nnet.use(Xtest)

fig = plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.plot(nnet.error_trace)
plt.xlabel("Epochs")
plt.ylabel("Likelihood")

plt.subplot(2, 2, 2)

for c in range(1, 4):
    mask = (T == c).flatten()
    plt.plot(X[mask, 0], X[mask, 1], 'o', markersize=6,  alpha=0.5,  color=colors[c-1])

plt.subplot(2, 2, 4)
plt.contourf(Xtest[:, 0].reshape((40, 40)), Xtest[:, 1].reshape((40, 40)),  predTest.reshape((40, 40)), 
             levels = [0.5, 1.99, 2.01, 3.5],  #    levels=(0.5, 1.5, 2.5, 3.5), 
             colors=colors);

st.pyplot(fig)

from matplotlib.colors import LightSource

fig = plt.figure( figsize=(6, 20))
ls = LightSource(azdeg=30,  altdeg=10)
white = np.ones((x.shape[0],  x.shape[1],  3))
red = white * np.array([1, 0.2, 0.2])
green = white * np.array([0.2, 1, 0.2])
blue = white * np.array([0.4, 0.4, 1])
colors = [blue, red,  green]

for c in range(3):
    ax = fig.add_subplot(3, 1, c+1, projection='3d')
    ax.view_init(azim = 180+40, elev = 60)
    Z = probs[:,  c].reshape(x.shape)
    rgb = ls.shade_rgb(colors[c],  Z,  vert_exag=0.1)
    ax.plot_surface(x, y, Z, 
                    rstride=1, cstride=1, linewidth=0,  antialiased=False, 
                    shade=True,  facecolors=rgb)
    ax.set_zlabel(r"$p(C="+str(c+1)+"|x)$")

st.pyplot(fig)
