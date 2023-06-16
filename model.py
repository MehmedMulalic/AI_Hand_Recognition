import Loading
import Layers
import numpy as np

# Load and Process Data
X, y = Loading.get_train_data()
X = X.reshape(-1, 2500)

keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

X_test, y_test = Loading.get_test_data()
X_test = X_test.reshape(-1, 2500)

# Creating Model Layers
model = Layers.Model()

model.add(Layers.Layer_Dense(2500, 256, weight_regularizer_l2=1e-4, bias_regularizer_l2=1e-4))
model.add(Layers.Activation_ReLU())
model.add(Layers.Layer_Dropout(0.5))
model.add(Layers.Layer_Dense(256, 64))
model.add(Layers.Activation_ReLU())
model.add(Layers.Layer_Dense(64, 10))
model.add(Layers.Activation_Softmax())

model.set(
    loss=Layers.Loss_CategoricalCrossentropy(),
    optimizer=Layers.Optimizer_Adam(learning_rate=0.01, decay=2e-4),
    accuracy=Layers.Accuracy_Categorical()
)

model.finalize()
model.train(X, y, validation_data=(X_test, y_test), epochs=130)
model.save('hand_gesture_model.model')