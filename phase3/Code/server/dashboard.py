import dash
from dash import dcc, html, callback
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Load the dataset
df = pd.read_csv('merged.csv')

# Create a list of unique neighborhood groups
neighborhood_groups = df['neighbourhood_group'].unique()

# Create the Dash app
app = dash.Dash()

# Define layout
app.layout = html.Div([
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
@app.callback(
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
                                         size='minimum_nights', hover_name='name', mapbox_style='carto-positron',
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


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True,port=9000)
