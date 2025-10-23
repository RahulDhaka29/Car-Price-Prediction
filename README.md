# Car Price Prediction (CarPrice AI)

A machine learning project that predicts the selling price of used cars based on their features. The project utilizes a dataset from CarDekho and employs a tuned Random Forest Regressor model deployed via a Flask web application.



## Features ✨

* **Regression Model:** Predicts the continuous value of a used car's selling price.
* **Feature Engineering:** Creates valuable features like `car_age` to improve model accuracy.
* **Model Comparison:** Evaluates multiple regression algorithms (Linear Regression, Random Forest, XGBoost) to select the best performer.
* **Hyperparameter Tuning:** Optimizes the selected Random Forest model using GridSearchCV for enhanced prediction accuracy.
* **Web Interface:** An interactive and user-friendly web application built with Flask and styled using Tailwind CSS, allowing users to input car details and receive price predictions.
* **Dynamic Dropdowns:** UI elements are populated directly from the dataset for a seamless user experience.

## Project Structure 📁
```
/car-price-prediction/ 
|-- app.py # Main Flask application logic 
|-- car_price_model.pkl # Saved final tuned Random Forest model 
|-- car_data.pkl # Saved final DataFrame structure (for prediction input) 
|-- car_prediction.ipynb # Jupyter notebook for EDA, model building, and tuning 
|-- CAR DETAILS FROM car_details.csv # Original dataset file (used for dropdowns in app.py) 
|-- README.md # This file 
|-- requirements.txt # List of Python dependencies 
|-- /templates/ 
|-- index.html # Main predictor page template 
|-- about.html # About page template 
|-- contact.html # Contact page template
```
---

*(Adjust file names like `.ipynb` and `.csv` if yours are different)*

## Dataset 📊

This project uses the **Vehicle Dataset from CarDekho**.

* **Source:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho)
* **Content:** The dataset contains details of used cars listed on CarDekho, including selling price, present price, kilometers driven, fuel type, seller type, transmission, owner history, and manufacturing year.
* **File Used:** `CAR DETAILS FROM CAR DEKHO.csv`. This file is required by `app.py` to populate the form dropdowns and should be present in the root directory alongside `app.py`.

---

## Setup and Installation ⚙️

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/car-price-predictor.git](https://github.com/your-username/car-price-predictor.git)
    cd car-price-predictor
    ```
    *(Replace `your-username` with your actual GitHub username)*
---
2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```
---
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Make sure you have created a `requirements.txt` file. See instructions below.)*

## How to Run 🚀

1.  **Ensure Model & Data Files Exist:** Make sure `car_price_model.pkl`, `car_data.pkl`, and `CAR DETAILS FROM CAR DEKHO.csv` are present in the project's root directory. If the `.pkl` files are missing, run the `car_prediction.ipynb` notebook to generate them.
2.  **Activate your virtual environment:** (See Setup step 2).
3.  **Run the Flask app:**
    ```bash
    python app.py
    ```
4.  **Open your web browser** and navigate to `http://127.0.0.1:5000`.
---
## Model Building Process (Summary) 🧠

The `car_prediction.ipynb` notebook details the model creation:
1.  **Data Loading & Initial Inspection:** Loaded the dataset (`CAR DETAILS FROM CAR DEKHO.csv`), checked data types, summary statistics, and for missing values (found none).
2.  **Exploratory Data Analysis (EDA):** Visualized the distribution of `selling_price`, analyzed categorical features, and plotted relationships between numerical features like `car_age` and `selling_price` using histograms, boxplots, and scatterplots.
3.  **Feature Engineering:** Created the `car_age` feature from the `year` column and extracted the `brand` from the `name` column. Dropped original unnecessary columns.
4.  **Data Preprocessing:** Applied One-Hot Encoding (`pd.get_dummies`) to convert categorical features (`fuel`, `seller_type`, `transmission`, `owner`, `brand`) into a numerical format suitable for modeling.
5.  **Model Selection:** Split the data into training and testing sets. Trained and evaluated three different regression models: `LinearRegression`, `RandomForestRegressor`, and `XGBRegressor`. Compared their performance using R-squared (R²) and Mean Absolute Error (MAE). Selected Random Forest as the best baseline model.
6.  **Hyperparameter Tuning:** Used `GridSearchCV` with 5-fold cross-validation to find the optimal hyperparameters for the `RandomForestRegressor`, further improving its R² score.
7.  **Final Model Training & Export:** Trained the optimized Random Forest model on the *entire* dataset using the best hyperparameters found during tuning. Saved the final trained model (`car_price_model.pkl`) and the final DataFrame structure (`car_data.pkl`) using `pickle`.
---
## Technologies Used 💻

* **Python:** Core programming language
* **Pandas:** Data manipulation, analysis, and one-hot encoding
* **NumPy:** Numerical computations
* **Matplotlib & Seaborn:** Data visualization for EDA
* **Scikit-learn:** Data splitting (`train_test_split`), model building (`LinearRegression`, `RandomForestRegressor`), model evaluation (`r2_score`, `mean_absolute_error`), and hyperparameter tuning (`GridSearchCV`)
* **XGBoost:** Used for model comparison (`XGBRegressor`)
* **Flask:** Micro web framework for the backend server
* **HTML & Tailwind CSS:** Frontend structure and styling
* **Jupyter Notebook:** For model development, EDA, and experimentation
* **Git & GitHub:** Version control and code hosting

---

## 👨‍💻 Author

Project created by **[Rahul Dhaka]**  
[LinkedIn](https://www.linkedin.com/in/rahul-dhaka-56b975289/),  [GitHub](https://github.com/RahulDhaka29)

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
**Creating `requirements.txt`:**

Before you upload your code, create a file named `requirements.txt` in your main project folder. This lists the Python libraries needed. In your terminal (with your virtual environment active), run:

```bash
pip freeze > requirements.txt