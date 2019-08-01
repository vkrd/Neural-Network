from Model import *

# test with XOR gate
feature_set = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
labels = np.array([[0], [0], [1], [1]])

# define network architecture
model = Model()

model.add_layer(Dense(2, name="i", activation="sig"))
model.add_layer(Dense(3, activation="sig"))
model.add_layer(Dense(1, name="o"))

model.train(feature_set, labels, epochs=10000, learning_rate=0.1, momentum=0.8)

print(model.predict(np.array([[0, 0], [1, 1], [1, 0], [0, 1]])))

model.save_model("model")
model.load_architecture("model.arc")
model.load_weights("model.weights")
