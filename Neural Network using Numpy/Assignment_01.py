import numpy as np

# Seeding for consistent production
np.random.seed(150)

fname = 'assign1_data.csv'
data = np.genfromtxt(fname, dtype='float', delimiter=',', skip_header=1)
X, y = data[:, :-1], data[:, -1].astype(int)
X_train, y_train = X[:400], y[:400]
X_test, y_test = X[400:], y[400:]

# Define Dense Layer
class DenseLayer:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
    def backward(self, dz):
        self.dweights = np.dot(self.inputs.T, dz)
        self.dbiases = np.sum(dz, axis=0, keepdims=True)
        self.dinputs = np.dot(dz, self.weights.T)

# Define ReLU activation function
class ReLu:
    def forward(self, z):
        self.z = z
        self.activity = np.maximum(0, z)
    def backward(self, dactivity):
        self.dz = dactivity.copy()
        self.dz[self.z <= 0] = 0.0

# Define Softmax activation function
class Softmax:
    def forward(self, z):
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        self.probs = e_z / e_z.sum(axis=1, keepdims=True)
        return self.probs
    def backward(self, dprobs):
        self.dz = np.empty_like(dprobs)
        for i, (prob, dprob) in enumerate(zip(self.probs, dprobs)):
            prob = prob.reshape(-1, 1)
            jacobian = np.diagflat(prob) - np.dot(prob, prob.T)
            self.dz[i] = np.dot(jacobian, dprob)

# Define Cross Entropy Loss
class CrossEntropyLoss:
    def forward(self, probs, oh_y_true):
        probs_clipped = np.clip(probs, 1e-7, 1 - 1e-7)
        loss = -np.sum(oh_y_true * np.log(probs_clipped), axis=1)
        return loss.mean(axis=0)
    def backward(self, probs, oh_y_true):
        batch_sz, n_class = probs.shape
        self.dprobs = -oh_y_true / probs
        self.dprobs = self.dprobs / batch_sz

# Define SGD Optimizer
class SGD:
   def __init__(self, learning_rate=1.0):
        self.learning_rate = learning_rate
   def update_params(self, layer):
        layer.weights -= self.learning_rate * layer.dweights
        layer.biases -= self.learning_rate * layer.dbiases

# Helper Functions
def predictions(probs):
    y_preds = np.argmax(probs, axis=1)
    return y_preds

def accuracy(y_preds, y_true):
    return np.mean(y_preds == y_true)

# Loss Function
loss_function = CrossEntropyLoss()

# Forward pass
def forward_pass(X):
    dense1.forward(X)
    activation1.forward(dense1.z)
    dense2.forward(activation1.activity)
    activation2.forward(dense2.z)
    output_layer.forward(activation2.activity)
    probs = output_activation.forward(output_layer.z)

    return probs

# Backward Pass
def backward_pass(probs, y_true, oh_y_true):
    loss_function.backward(probs, oh_y_true)
    output_activation.backward(loss_function.dprobs)
    output_layer.backward(output_activation.dz)
    activation2.backward(output_layer.dinputs)
    dense2.backward(activation2.dz)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dz)

# Initialize the network and set hyperparameters
epochs = 9
batch_sz = 32
n_neurons_hidden1 = 4
n_neurons_hidden2 = 8
n_inputs = 3
n_class = 3

# Create the layers of the neural network
dense1 = DenseLayer(n_inputs, n_neurons_hidden1)
activation1 = ReLu()
dense2 = DenseLayer(n_neurons_hidden1, n_neurons_hidden2)
activation2 = ReLu()
output_layer = DenseLayer(n_neurons_hidden2, n_class)
output_activation = Softmax()
crossentropy = CrossEntropyLoss()
optimizer = SGD()

# Training loop
for epoch in range(epochs):
    n_batch = len(X_train) // batch_sz
    for batch_i in range(n_batch):
        # Get a mini-batch of data from X_train and y_train
        batch_X = X_train[batch_i * batch_sz:(batch_i + 1) * batch_sz]
        batch_y = y_train[batch_i * batch_sz:(batch_i + 1) * batch_sz]

        # One-hot encode y_true
        oh_batch_y = np.eye(n_class)[batch_y]

        # Forward pass
        probs = forward_pass(batch_X)

        # Loss
        loss = crossentropy.forward(probs, oh_batch_y)

        # Print accuracy and loss
        y_preds = predictions(probs)
        acc = accuracy(y_preds, batch_y)
        print(f'Epoch {epoch+1}, Batch {batch_i+1}, Loss: {loss:.4f}, Accuracy: {acc:.4f}')

        # Backward pass
        backward_pass(probs, batch_y, oh_batch_y)

        # Update weights and biases
        optimizer.update_params(dense1)
        optimizer.update_params(dense2)
        optimizer.update_params(output_layer)

probs = forward_pass(X_test)
y_preds = predictions(probs)
test_accuracy = accuracy(y_preds, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")
