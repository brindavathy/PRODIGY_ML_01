from django.shortcuts import render
from .utils import predict_price

def predict_view(request):
    prediction = None
    error_message = None
    
    if request.method == 'POST':
        try:
            feature1 = request.POST.get('feature1')
            feature2 = request.POST.get('feature2')
            feature3 = request.POST.get('feature3')
            feature4 = request.POST.get('feature4')
            # Validate inputs
            if not (feature1 and feature2 and feature3 and feature4):
                raise ValueError("All features are required.")
            
            feature1 = float(feature1)
            feature2 = float(feature2)
            feature3 = float(feature3)
            feature4 = float(feature4)
            
            features = [feature1, feature2, feature3, feature4]
            
            # Assuming predict_price returns a list or array and the first element is the prediction
            prediction = predict_price(features)[0]
            prediction = round(prediction, 2)
            
        except (ValueError, TypeError) as e:
            error_message = "Invalid input: please ensure all features are numbers."
            print("Error:", e)

    return render(request, 'form.html', {'prediction': prediction, 'error_message': error_message})
