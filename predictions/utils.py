import pickle
import numpy as np

def predict_price(features):
    # Load the trained model (ensure the path to your model is correct)
    with open('predictions/model/house_price_model.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Convert features to numpy array
    features_array = np.array(features).reshape(1, -1)
    
    # Predict using the model
    prediction = model.predict(features_array)
    return prediction
