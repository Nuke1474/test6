import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

logic_gates = {
    "AND": {
        "inputs": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "outputs": np.array([[0], [0], [0], [1]])
    },
    "OR": {
        "inputs": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "outputs": np.array([[0], [1], [1], [1]])
    },
    "XOR": {
        "inputs": np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
        "outputs": np.array([[0], [1], [1], [0]])
    }
}


for gate_type, data in logic_gates.items():
    inputs = data["inputs"]
    outputs = data["outputs"]
    model = Sequential([
        Dense(4, input_dim=2, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.01), loss='binary_crossentropy', metrics=['accuracy'])
    print(f"\nTraining for {gate_type} gate...")
    model.fit(inputs, outputs, epochs=500, verbose=0)
    predictions = model.predict(inputs)
    
    print(f"\nResults for {gate_type} gate:")
    
    for i in range(len(inputs)):
        predicted_output = round(predictions[i][0])
        true_output = outputs[i][0]
        print(f"Input: {inputs[i]} => Predicted Output: {predicted_output}, True Output: {true_output}")
        







