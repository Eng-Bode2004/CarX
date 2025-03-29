import gradio as gr
import joblib
import pandas as pd

# Load the trained Random Forest model
model_path = "random_forest_model.pkl"  # Update this path if needed
model = joblib.load(model_path)

# Define the feature names (update these based on your dataset features)
feature_names = ["Year", "Mileage", "Engine_Size", "Horsepower", "Fuel_Type", "Transmission", "Brand"]

def predict_price(year, mileage, engine_size, horsepower, fuel_type, transmission, brand):
    # Convert input to DataFrame
    input_data = pd.DataFrame([[year, mileage, engine_size, horsepower, fuel_type, transmission, brand]],
                              columns=feature_names)
    # Make prediction
    prediction = model.predict(input_data)[0]
    return f"Estimated Car Price: ${prediction:,.2f}"

# Create Gradio interface
iface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Year"),
        gr.Number(label="Mileage"),
        gr.Number(label="Engine Size"),
        gr.Number(label="Horsepower"),
        gr.Dropdown(["Petrol", "Diesel", "Electric", "Hybrid"], label="Fuel Type"),
        gr.Dropdown(["Manual", "Automatic"], label="Transmission"),
        gr.Dropdown(["Kia", "Chery", "Fiat", "Hyundai", "BMW", "Chevrolet"], label="Brand")  # Replace with actual brands in dataset
    ],
    outputs=gr.Textbox(label="Prediction"),
    title="Car Price Prediction",
    description="Enter car details to predict its price using the Random Forest model."
)


# Launch the GUI
if __name__ == "__main__":
    iface.launch()
