from flask import Flask, request, jsonify, redirect, url_for
from flask import render_template
import numpy as np
from flask_cors import CORS
import util
import dash
import pickle
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import pickle
from flask import request, render_template



# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set the path to the static directory
app._static_folder = os.path.abspath("./templates")

with open('./artifacts/knn_model.pickle', 'rb') as file:
        model_data = pickle.load(file)

# Initialize Dash app within the Flask app
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dashboard/')
dash_app.title = 'Airbnb NYC 2019 Dashboard'

# Load the dataset
df = pd.read_csv('data.csv')

# Create a list of unique neighborhood groups
neighborhood_groups = df['neighbourhood_group'].unique()

# Define layout for the Dash app
dash_app.layout = html.Div([
    html.H1("Airbnb NYC 2019 Dashboard"),

    # Dropdown for Neighborhood Groups
    dcc.Dropdown(
        id='neighborhood-dropdown',
        options=[{'label': group, 'value': group} for group in neighborhood_groups],
        value=neighborhood_groups[0],
        style={'width': '50%'}
    ),

    # Graphs
    dcc.Graph(id='price-distribution'),
    dcc.Graph(id='map-listings'),
    dcc.Graph(id='room-type-distribution'),
    dcc.Graph(id='price-vs-reviews'),
])

# Define callback to update graphs based on dropdown selection
@dash_app.callback(
    [Output('price-distribution', 'figure'),
     Output('map-listings', 'figure'),
     Output('room-type-distribution', 'figure'),
     Output('price-vs-reviews', 'figure')],
    [Input('neighborhood-dropdown', 'value')]
)
def update_graphs(selected_neighborhood):
    filtered_df = df[df['neighbourhood_group'] == selected_neighborhood]

    # Distribution of Prices
    price_distribution_fig = px.histogram(filtered_df, x='price', nbins=30, title='Distribution of Prices',
                                          range_y=[0, 500],  # Set the desired range for the y-axis
                                          range_x=[0, 1000])  # Set the desired range for the x-axis

    # Map of Airbnb listings
    map_listings_fig = px.scatter_mapbox(filtered_df, lat='latitude', lon='longitude', color='price',
                                         size='price', hover_name='name', mapbox_style='carto-positron',
                                         color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10)

    # Room Type Distribution
    room_type_distribution_fig = px.bar(filtered_df['room_type'].value_counts(),
                                        x=filtered_df['room_type'].value_counts().index,
                                        y=filtered_df['room_type'].value_counts().values,
                                        labels={'x': 'Room Type', 'y': 'Count'},
                                        title='Room Type Distribution')

    # Price vs. Number of Reviews
    price_vs_reviews_fig = px.scatter(filtered_df, x='accommodates', y='price', trendline='ols',
                                      labels={'x': 'accommodates', 'y': 'Price'},
                                      title='Price vs. accommodates')

    return price_distribution_fig, map_listings_fig, room_type_distribution_fig, price_vs_reviews_fig


@app.route('/')
def home():
    return render_template('app.html')

# Flask route to serve the Dash app
@app.route('/dashboard/')
def index():
    return dash_app.index()


@app.route('/predict_home_price', methods=['POST'])
def predict_home_price():
    neighborhood_group = request.form['neighborhood_group']
    bedrooms = int(request.form['bedrooms'])
    beds = int(request.form['beds'])
    accommodates = int(request.form['accommodates'])
    room_type = request.form['room_type']
    minimum_nights = int(request.form['minimum_nights'])
    availability_365 = int(request.form['availability_365'])

    # Use these values in your price estimation logic
    # Replace the logic below with your actual price prediction code
    estimated_price = util.evaluate_knn_model(neighborhood_group, bedrooms, beds, accommodates, room_type, minimum_nights, availability_365)

    # Returning the estimated price as JSON response
    response = jsonify({'estimated_price': estimated_price})
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    # Load your model
    with open('./artifacts/knn_model.pickle', 'rb') as file:
        model_data = pickle.load(file)
        knn_model_retrain = model_data['model']

    # Load the uploaded dataset
    dataset = request.files.get('file')

    def preprocess_data(user_data):
    # Handle missing values with imputation for numeric columns
        numeric_cols = user_data.select_dtypes(include=['number']).columns
        imputer_numeric = SimpleImputer(strategy='mean')
        user_data[numeric_cols] = imputer_numeric.fit_transform(user_data[numeric_cols])

        # Handle missing values for non-numeric columns (e.g., using a constant)
        non_numeric_cols = user_data.select_dtypes(exclude=['number']).columns
        imputer_non_numeric = SimpleImputer(strategy='constant', fill_value='missing')
        user_data[non_numeric_cols] = imputer_non_numeric.fit_transform(user_data[non_numeric_cols])

        # Encode categorical columns
        categorical = user_data.select_dtypes(include=['object', 'category']).columns
        for col in categorical:
            user_data[col] = LabelEncoder().fit_transform(user_data[col])

        X = user_data.drop(columns=['price'])
        y = user_data['price']

        columns = X.columns
        scaler = StandardScaler()
        X[columns] = scaler.fit_transform(X[columns])

        return X, y

    if dataset:
        # Read the dataset
        user_data = pd.read_csv(dataset)

        # Preprocess the dataset
        X, y = preprocess_data(user_data)

        # Retrain or fine-tune your model with new data
        knn_model_retrain.fit(X, y)

        # Make predictions
        y_pred = knn_model_retrain.predict(X)

        # Calculate evaluation metrics
        mse = mean_squared_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y, y_pred)
        print(mse,mae,rmse)
        evaluation_metrics = {
            "Mean Squared Error": mse,
            "Mean Absolute Error": mae,
            "Root Mean Squared Error": rmse,
            "R2 Score": r2
        }

        return render_template('result.html', metrics=evaluation_metrics)

if __name__ == "__main__":
    util.load_saved_artifacts()
    print("Starting the server...")
    app.run(debug=True)
