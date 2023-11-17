from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Example model architecture
model = Sequential([
    Dense(64, activation='relu', input_shape=(100,)),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])

# Compile and train the model (Example training)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
X_train = np.random.random((1000, 100))
y_train = np.random.randint(10, size=(1000,))
model.fit(X_train, y_train, epochs=10)

# Save the model weights to 'weights.h5'
model.save_weights('weights.h5')
