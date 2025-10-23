# --- Step 1: Import Necessary Libraries ---
from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np

# --- Step 2: Initialize the Flask App ---
app = Flask(__name__)

# --- Step 3: Load the Saved Model and Data Structures ---
try:
    model = pickle.load(open('car_price_model.pkl', 'rb'))
    car_data = pickle.load(open('car_data.pkl', 'rb'))
    print("Model and data loaded successfully.")
except FileNotFoundError:
    print("ERROR: One or more .pkl files not found. Make sure 'car_price_model.pkl' and 'car_data.pkl' are in the directory.")
    exit()

# --- Step 4: Prepare Data for Dropdowns (The Robust Way) ---
# This is the new, improved method. We extract options directly from our model's data.
all_columns = car_data.columns
brands = sorted(list(set([col.split('_')[1] for col in all_columns if col.startswith('brand_')])))
try:
    original_df = pd.read_csv('car_details.csv')
    original_df['brand'] = original_df['name'].str.split().str[0]
    all_brands = sorted(original_df['brand'].unique())
    brands = all_brands
except FileNotFoundError:
    print("Warning: 'car_details.csv' not found. Brand list might be incomplete.")
    if 'Maruti' not in brands: brands.insert(0, 'Maruti')

fuels = sorted(list(set([col.split('_')[1] for col in all_columns if col.startswith('fuel_')])))
if 'Diesel' not in fuels: fuels.insert(0, 'Diesel')

sellers = sorted(list(set([col.split('_')[1] for col in all_columns if col.startswith('seller_type_')])))
if 'Dealer' not in sellers: sellers.insert(0, 'Dealer')

transmissions = sorted(list(set([col.split('_')[1] for col in all_columns if col.startswith('transmission_')])))
if 'Automatic' not in transmissions: transmissions.insert(0, 'Automatic')

owners = sorted(list(set([col.split('_')[1] for col in all_columns if col.startswith('owner_')])))
if 'First Owner' not in owners: owners.insert(0, 'First Owner')


# --- Step 5: Define Web Routes (Pages) ---
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = ""
    
    if request.method == 'POST':
        try:
            # --- Get all inputs from the user form ---
            car_age = int(request.form.get('car_age'))
            km_driven = int(request.form.get('km_driven'))
            owner = request.form.get('owner')
            brand = request.form.get('brand')
            fuel = request.form.get('fuel')
            seller_type = request.form.get('seller_type')
            transmission = request.form.get('transmission')
            
            # --- ** THE FIX IS HERE ** ---
            # --- Prepare the input data for the model ---
            # We must exactly replicate the feature engineering from the notebook.

            # 1. Get the list of columns the model expects.
            X_columns = car_data.drop(columns=['selling_price']).columns
            
            # 2. Create a DataFrame for our input, filled with zeros.
            input_data = pd.DataFrame(np.zeros((1, len(X_columns))), columns=X_columns)
            
            # 3. Fill in the numerical values, including the engineered feature.
            input_data['car_age'] = car_age
            
            # Calculate km_per_year, handling the case where age is 0.
            car_age_safe = car_age if car_age > 0 else 1
            km_per_year = km_driven / car_age_safe
            if 'km_per_year' in input_data.columns:
                 input_data['km_per_year'] = km_per_year
            
            # 4. Set the '1' for the selected categorical features.
            if f'brand_{brand}' in input_data.columns:
                input_data[f'brand_{brand}'] = 1
            if f'fuel_{fuel}' in input_data.columns:
                input_data[f'fuel_{fuel}'] = 1
            if f'seller_type_{seller_type}' in input_data.columns:
                input_data[f'seller_type_{seller_type}'] = 1
            if f'transmission_{transmission}' in input_data.columns:
                input_data[f'transmission_{transmission}'] = 1
            if f'owner_{owner}' in input_data.columns:
                input_data[f'owner_{owner}'] = 1

            # --- Make the Prediction ---
            prediction = model.predict(input_data)
            
            # Format the prediction for display (e.g., "₹ 5.75 Lakhs").
            prediction_text = f"Predicted Price: ₹ {prediction[0] / 100000:.2f} Lakhs"

        except Exception as e:
            prediction_text = f"An error occurred: {e}"

    return render_template('index.html', 
                           prediction_text=prediction_text,
                           brands=brands,
                           fuels=fuels,
                           sellers=sellers,
                           transmissions=transmissions,
                           owners=owners)

# Route for the 'About' page
@app.route('/about')
def about():
    return render_template('about.html')

# Route for the 'Contact' page
@app.route('/contact')
def contact():
    return render_template('contact.html')

# --- Step 6: Run the App ---
if __name__ == '__main__':
    app.run(debug=True)

