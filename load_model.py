from tensorflow import keras

# Load the model
model = keras.models.load_model(r'E:\mashkoor ali\MCA\final year project mca\virtual_env\env\model.keras')

# Print the model summary
model.summary()
