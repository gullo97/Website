#%%
import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import seaborn as sns
import os
import pickle
import plotly.graph_objects as go
# %%

class SimpleNN(nn.Module):
    def __init__(self, input_dim, output_dim, mode='classification'):
        super(SimpleNN, self).__init__()
        self.mode = mode
        
        self.norm = nn.BatchNorm1d(input_dim)
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, output_dim)
        
        self.relu = nn.ReLU()
        
        if self.mode == 'classification':
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.norm(x)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        if self.mode == 'classification':
            x = self.softmax(x)
        return x
    
    def predict(self, x):
        self.eval()
        if isinstance(x, pd.DataFrame):
            x = torch.tensor(x.values, dtype=torch.float32)
        
        with torch.no_grad():
            predictions = self.forward(x)
            if self.mode == 'classification':
                _, predicted = torch.max(predictions.data, 1)
                return predicted.numpy()
            else:
                return predictions.numpy()
    def forward_logits(self, x):
        x = self.norm(x)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

class MapperNN(nn.Module):
    '''This model augments the coordinates by calculating the squares and cubes of the coordinates and a few sine and cosine transformations of the coordinates.
        This allows the model to learn the shape of the vulnerability map.
    '''
    def __init__(self, input_dim, output_dim, mode='classification', n_harmonics=5, omega_max=10):
        super(MapperNN, self).__init__()
        self.mode = mode
        self.omegas = nn.Parameter(torch.randn(n_harmonics, 7) * omega_max) 
        self.amplitudes = nn.Parameter(torch.randn(n_harmonics, 7))
        # self.amplitudes = nn.Parameter(torch.ones(n_harmonics, 7)).requires_grad_(False)
        self.phases_sin = nn.Parameter(torch.randn(n_harmonics, 7))
        self.phases_cos = nn.Parameter(torch.randn(n_harmonics, 7))
        self.biases_sin = nn.Parameter(torch.zeros(n_harmonics, 7))
        self.biases_cos = nn.Parameter(torch.zeros(n_harmonics, 7))
        
        
        
        self.norm = nn.BatchNorm1d(input_dim)
        self.layer1 = nn.Linear(input_dim + 14 * n_harmonics + 2 * 7, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 80)
        self.layer4 = nn.Linear(80, 64)
        self.layer5 = nn.Linear(64, output_dim)
        
        self.activation_f = nn.ELU()

        if self.mode == 'classification':
            self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.norm(x)
        coordinates = x[:, :7] # this is a tensor of shape (batch_size, 2)
        onehot = x[:, 7:] # this is a tensor of shape (batch_size, n_onehot)
        
        # calculate the squares and cubes of the coordinates
        squares = coordinates ** 2 # this is a tensor of shape (batch_size, 2)
        cubes = coordinates ** 3    # this is a tensor of shape (batch_size, 2)
        
        coordinates = coordinates.unsqueeze(1) # this is a tensor of shape (batch_size, 1, 2)
        
        # calculate sine and cosine transformations of the coordinates with different omegas and amplitudes
        sines = torch.sin((coordinates + self.biases_sin) * self.omegas + self.phases_sin) * self.amplitudes # this is a tensor of shape (batch_size, n_harmonics, 2), where coordinates is broadcasted to (batch_size, n_harmonics, 2), this creates n_harmonics tensors of shape (batch_size, 2)
        cosines = torch.cos((coordinates + self.biases_cos) * self.omegas + self.phases_cos) * self.amplitudes # this is a tensor of shape (batch_size, n_harmonics, 2), where coordinates is broadcasted to (batch_size, n_harmonics, 2), this creates n_harmonics tensors of shape (batch_size, 2)
        
        sines = sines.reshape(sines.shape[0], -1) # this is a tensor of shape (batch_size, 2 * n_harmonics)
        cosines = cosines.reshape(cosines.shape[0], -1) # this is a tensor of shape (batch_size, 2 * n_harmonics)
        
        coordinates = coordinates.squeeze(1) # this is a tensor of shape (batch_size, 2)
        # concatenate the coordinates, squares, cubes, sines and cosines
        x = torch.cat([coordinates, squares, cubes, sines, cosines, onehot], dim=1) 
        
        
        x = self.activation_f(self.layer1(x))
        x = self.activation_f(self.layer2(x))
        x = self.activation_f(self.layer3(x))
        x = self.activation_f(self.layer4(x))
        x = self.layer5(x)

        if self.mode == 'classification':
            x = self.softmax(x)
        return x
    
    def predict(self, x):
        self.eval()
        if isinstance(x, pd.DataFrame):
            x = torch.tensor(x.values, dtype=torch.float32)
        
        with torch.no_grad():
            predictions = self.forward(x)
            if self.mode == 'classification':
                _, predicted = torch.max(predictions.data, 1)
                return predicted.numpy()
            else:
                return predictions.numpy()
    def forward_logits(self, x):
        x = self.norm(x)
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

#%%

# st.write('This app does things')

@st.cache
def load_data():
    df = pd.read_csv('app/dataframe.csv')
    df = df.drop(columns=['Unnamed: 0'])
    return df

@st.cache
def load_event_data():
    df = pd.read_csv('app/aquila2009_caratteristiche_evento.csv', sep = ';')
    # df = df.drop(columns=['Unnamed: 0'])
    return df

@st.cache
def load_eq_coords():
    eq_coords = event_df[['lat_epicentro', 'lon_epicentro']].rename(columns={'lat_epicentro': 'Latitude', 'lon_epicentro': 'Longitude'})
    return eq_coords

@st.cache
def load_model(model_name='Simple NN'):
    if model_name == 'Simple NN':
        model = SimpleNN(81, 4)
        model.load_state_dict(torch.load('app/Simple NN/simple_nn.pth'))
    elif model_name == 'Harmonic Mapper':
        model = MapperNN(81, 4, n_harmonics=10)
        model.load_state_dict(torch.load('app/Harmonic Mapper/10har_mapper_nn.pth'))
    return model.eval()


@st.cache
def load_vulnerability_map_and_show(model_name = 'Simple NN'):
    path = 'app/'+model_name+'/vulnerability_map.html'
    with open(path, 'r') as file:
        map_html = file.read()
    return map_html

@st.cache
def load_image(path):
    return plt.imread(path)

@st.cache
def plot_on_map_plotly(grid, min_lat, max_lat, min_lon, max_lon, eq_df, dataframe, n_samples=1000, zoom_factor=5, zoom_center=(41.9, 12.5)):
    # Normalize the grid
    normed_grid = (grid - grid.min()) / (grid.max() - grid.min())
    # flip normed_grid vertically
    normed_grid = np.flip(normed_grid, axis=0)

    # Create latitudes and longitudes for the grid
    lats = np.linspace(min_lat, max_lat, grid.shape[0])
    lons = np.linspace(min_lon, max_lon, grid.shape[1])

    # Create a DataFrame for the grid
    grid_data = pd.DataFrame([(lat, lon, normed_grid[i, j]) for i, lat in enumerate(lats) for j, lon in enumerate(lons)], columns=['Latitude', 'Longitude', 'Intensity'])
    
    # Create a Plotly figure
    fig = go.Figure()

    # Add the grid as a density mapbox layer
    fig.add_trace(go.Densitymapbox(lat=grid_data['Latitude'], lon=grid_data['Longitude'], z=grid_data['Intensity'],
                                colorscale='Plasma', radius=10, opacity=0.5, # for green to red, use colorscale='YlOrRd', for blue to red, use colorscale='Bluered', for blue to yellow, use colorscale='Viridis'
                                reversescale=False,
                                # other colorscales: 'Viridis', 'Cividis', 'Bluered_r', 'RdBu_r', 'RdBu', 'Picnic', 'Portland', 'Jet', 'Hot', 'Blackbody', 'Earth', 'Electric', 'YlOrRd', 'YlOrBr', 'Reds', 'Blues', 'Greens', 'YlGnBu', 'YlGn', 'Rainbow', 'RdBu', 'Picnic', 'Rainbow', 'Portland', 'Jet', 'Hot', 'Blackbody', 'Earth', 'Electric', 'YlOrRd', 'YlOrBr', 'Reds', 'Blues', 'Greens', 'YlGnBu', 'YlGn'
                                name='Vulnerability Score', 
                                hovertemplate='Vulnerability: %{z:.2f}<br>Latitude: %{lat:.3f}<br>Longitude: %{lon:.3f}<extra></extra>',
                                colorbar=dict(x=-0.1, xpad=10))) # xpad to adjust the position of the colorbar
    # Sample the building coordinates
    sample_data = dataframe.sample(n_samples)

    # Scatter plot for each damage level
    damage_keys = ['No Damage', 'Light Damage', 'Moderate Damage', 'Severe Damage']
    for damage_level in [0, 1, 2, 3]:
        data = sample_data[sample_data['Vertical Structural Damage'] == damage_level]
        colors = data['Vertical Structural Damage'].map({0: 'green', 1: 'yellow', 2: 'orange', 3: 'red'})
        fig.add_trace(go.Scattermapbox(lon=data['Longitude'], 
                                        lat=data['Latitude'],
                                        hovertemplate='Damage Level: ' + damage_keys[damage_level] + '<br>Latitude: %{lat:.3f}<br>Longitude: %{lon:.3f}<extra></extra>',
                                        mode='markers', marker=dict(color=colors, size=6), name=damage_keys[damage_level]))

    # Plot epicenters
    fig.add_trace(go.Scattermapbox(lon = [float(x) for x in eq_df['lon_epicentro'].values], lat = [float(y) for y in eq_df['lat_epicentro'].values],
                                    mode='markers+text',
                                    text=['Magnitude: ' + str(value) for value in eq_df['magnitudo_mw'].values], textposition='top center', textfont=dict(color='blue', size=8), marker=dict(color='blue',size=8), name='Epicenters'))# alternative modes are 'markers', 'text', 'lines', 'markers+text', 'lines+markers', 'lines+text', 'lines+markers+text'
    # Set up the map layout
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(
            center=dict(lat=zoom_center[1], lon=zoom_center[0]),
            zoom=zoom_factor
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        legend=dict(x=1, y=0.5, xanchor='left', yanchor='middle')  # Adjust legend position here

    )

    return fig

def create_one_hot_from_selections(df, selections, latitude, longitude):
    # create a one-hot encoded vector from the selections
    one_hot = np.zeros(81)
    for feature, selection in selections.items():
        one_hot[df.columns.get_loc(selection)] = 1
    one_hot[0] = latitude
    one_hot[1] = longitude
    for i in range(len(eq_coords)):
        one_hot[2+i] = geodesic((latitude, longitude), (eq_coords.iloc[i]['Latitude'], eq_coords.iloc[i]['Longitude'])).km
    return one_hot

st.title('Machine learning for earthquake damage prediction and vulnerability assessment')
st.write('This app is a demo of the machine learning models developed for the paper "Seismic Vulnerability Assessment at Urban Scale by Means of Machine Learning Techniques" by Guglielmo Ferranti et. al, currently under review for publication on MDPI Buildings and also available at https://www.preprints.org/manuscript/202312.1304/v2. Here, we explain the "Simple NN" model used in the paper and build upon it to improve performance and interpretability with the "Harmonic Mapper" model.')

st.write('The study leverages data from the 2009 L\'Aquila earthquake in Italy to train a machine learning model to predict the damage level of a building given its characteristics, position and the earthquake characteristics. The model is at first used to interpolate a vulnerability map tailored to the seismic event and then, using a similar approach, to extrapolate an "A-Posteriori" vulnerability score for each building feature that minimizes its dependency on the specific seismic event and can, in principle, be useful outside of the specific event studied.')
df = load_data()

st.write('The dataset developed by Eucentre (European Center for Training and Research in Seismic Engineering, http://egeos.eucentre.it/danno_osservato/web/danno_osservato) contains a mix of categorical and numerical features. It is important to note that the original data is not shown on this app exept in aggregate form in accordance with the terms stated in the website. The categorical features are one-hot encoded to give a total of 81 features for each building. Here we report a synthetic example of the dataset that originally contains about 60000 samples')
st.write(df.head())
# st.write('Sample size: ' + str(len(df)))

event_df = load_event_data()
st.write('The data is augmented by precalculating the distance of each building from the main epicenters of the earthquake.')
st.write(event_df.head())

st.title('Model selection and performance')
st.write('Select a neural network model to see its performance on the validation set, interpolate a vulnerability map and play with damage prediction.')
model_names = ['Simple NN', 'Harmonic Mapper']
model_name = st.selectbox('Model', model_names)
# model_name = st.sidebar.selectbox('Model', model_names)
st.write('First we take a look at the performance of the model on the validation dataset')
st.image('app/'+model_name+'/report.png') 


st.write('We can see the model structure here, noting that the dataset and model are small enough to pass the whole validation dataset (containing 11628 samples) through the model at once.')

if model_name == 'Harmonic Mapper':
    st.write('The Harmonic Mapper neural network augments the numeric values in the data by calculating the squares, cubes and a selectable number of sine and cosine transformations of the coordinates and distances from epicenters, for each building. The parameters (phase, amplitude, omega and bias) of these sin/cos transformations are learned by the model during training to allow the model to better learn the shape of the vulnerability map.')
elif model_name == 'Simple NN':
    st.write('The Simple NN model is a simple fully connected neural network with 2 hidden layers of 128 and 64 neurons respectively. The input is a one-hot vector of length 81 and the output is a vector of length 4, representing the probabilities for a building to expect each of the 4 damage levels.')
# open svg file
with open('app/'+model_name+'/model.onnx.svg', 'rb') as f:
    svg = f.read()
    # st.write(svg.decode('utf-8'))
    st.image(svg.decode('utf-8'))
    




st.title('Vulnerability Map')

if model_name == 'Simple NN':
    st.write('Note: Select "Harmonic Mapper" in the model selector (above in this page) for the best results.')

st.write('We interpolate a continous vulnerability map by taking a diverse sample of buildings from the dataset and ask the model to predict the damage of these "dummy" buildings if they were positioned at each point on a grid. The grid is then colored according to the average damage predicted by the model. The grid is normalized to have values between 0 and 1, where 0 is the minimum vulnerability and 1 is the maximum. This approach aims to average out the effect of all possible feature combination and extract the weight assigned by the model to just the position of a building.')
st.write('The map also shows a very small sample of buildings\' coordinates (colored by damage level) and the epicenters of the earthquake (blue dots). This data is in aggregate form and does not allow to identify specific buildings or their features.')

# map_html = load_vulnerability_map_and_show(model_name=model_name) # <----------- Plotly map
# # Use the Streamlit components to render HTML
# components.html(map_html, height=500) 
grid_path = 'app/'+model_name+'/vulnerability_grid.npy'
grid = np.load(grid_path)
min_lat = 41.7158661414168
max_lat = 42.7362168
min_lon = 13.0
max_lon = 14.2035408
eq_df = event_df[['lat_epicentro', 'lon_epicentro', 'magnitudo_mw']]
fig = plot_on_map_plotly(grid, min_lat, max_lat, min_lon, max_lon, eq_df, df, n_samples=1000, zoom_factor=8, zoom_center=(13.4,42.3))
st.write(fig)

if model_name == 'Simple NN':
    st.write('The vulnerability map shows that the "Simple NN" has trouble fitting the complex shape of the seismic event. This is because the model is not large enough to learn the shape of the map and instead learns to predict the damage level based only on the features of the building. This causes the model to overfit the data and is a problem for the "A-Posteriori" vulnerability score, which is meant to be used on new buildings and locations.')

st.write('The Harmonic Mapper model is able to learn a detailed vulnerability map: this allows it to assign the correct portion of vulnerability to the coordinates of a building and thus makes it the better choice for A-Posteriori vulnerability estimation. This capability is critical because we want the model to be able to distinguish the intrinsic vulnerability of building features from their position relative to an earthquake, thus using the position as a bias for prediction. The better the model is at interpolating a map (and maintaning satisfactory F1 score on validation), the more confident we are in its ability to evaluate intrinsic vulnerability of specific features.')

# ########################################################
# # Assuming 'fig' is your Plotly figure object

# # Callback function to capture click events
# def on_click(trace, points, selector):
#     # Check if there are clicked points
#     if points.point_inds:
#         # Getting the index of the clicked point
#         index = points.point_inds[0]
#         clicked_lat = points.ys[index]
#         clicked_lon = points.xs[index]

#         # Updating Streamlit state
#         st.session_state.clicked_lat = clicked_lat
#         st.session_state.clicked_lon = clicked_lon

# # Example of adding click event to the first trace (modify as needed for your specific trace)
# fig.data[0].on_click(on_click)

# # Display the Plotly figure in Streamlit
# # st.plotly_chart(fig)

# # Displaying clicked coordinates
# if 'clicked_lat' in st.session_state and 'clicked_lon' in st.session_state:
#     st.write(f"Clicked Coordinates: Latitude - {st.session_state.clicked_lat}, Longitude - {st.session_state.clicked_lon}")
# ########################################################

st.title('Building simulation and vulnerability prediction')
st.write('Use the feature selection to simulate a building and predict the damage.')

eq_coords = load_eq_coords()
# st.write(eq_coords)

model = load_model(model_name=model_name)



# get names of macro features
categorical_features = ['Position in Complex',
    'Horizontal Structure',
    'Average Floor Height',
    'Number of Basement Floors',
    'Vertical Structure',
    'Number of Floors',
    'Chains or Beams',
    'Floor Area',
    'Isolated Pillars',
    'Construction or Restructuring',
    'Slope Morphology']

pag_col1, pag_col2 = st.columns(2)

#################################### Feature selection ####################################
selections = {}
for feature in categorical_features:
    feature_columns = [col for col in df.columns if feature in col]
    # remove string 'feature'+ ' :' from column names
    option_names = [col.split(':')[1] for col in feature_columns]
    selection = pag_col1.selectbox(feature, option_names)
    selections[feature] = feature + ':' + selection
    
latitude = pag_col2.slider('Latitude', min_value=41.0, max_value=44.0, value=42.34)
longitude = pag_col2.slider('Longitude', min_value=12.0, max_value=15.0, value=13.38)
###########################################################################################

# st.write(selections)



one_hot = create_one_hot_from_selections(df, selections, latitude, longitude)
# st.write('This is the one-hot vector for debugging')
# st.write(one_hot)

# give the one-hot vector to the model
prediction = model.predict(torch.tensor(one_hot, dtype=torch.float32).unsqueeze(0))
probs = model(torch.tensor(one_hot, dtype=torch.float32).unsqueeze(0)).detach().numpy()



# st.write(prediction)
# st.write(probs)
# bar plot of probabilities
fig, ax = plt.subplots()
ax.bar(['No Damage', 'Light Damage', 'Moderate Damage', 'Severe Damage'], probs[0])
# rotate x labels
plt.xticks(fontsize=9)
ax.set_ylabel('Probability')
# ax.set_xlabel('Damage Level')
pag_col2.pyplot(fig)
st.title('A-Posteriori Vulnerability Score')
st.write('The second step is dedicated to evaluating the predictive power of our models and establishing the advantages of our vulnerability scoring method. In particular, we introduce an innovative technique to derive an a-posteriori vulnerability score for each structural feature of the buildings in our dataset. In the following we show how it works:')

st.markdown("""
1. **Creation of Dummy Buildings**: A large sample of buildings are created, identical to real ones (including position) except for one static feature of interest. For example, simulating all buildings to have exactly two floors.
2. **Model Predictions**: These dummy buildings are fed into our Neural Network (and Random Forest) models, which then predict damage, considering the static feature as a key variable.
3. **Derivation of A-Posteriori Vulnerability Score**: By examining the damage predictions across dummy buildings with the constant feature, we derive an average damage score, representing the vulnerability attributed to that feature (like having two floors).
""", unsafe_allow_html=True)
st.write('The following plot shows the A-Posteriori vulnerability score for each feature calculated by the Harmonic Mapper ANN (in red), a Random Forest Classifier (in green) and their average (blue). The higher the score, the more vulnerable the feature is.')

scores_image = load_image('app/a_posteriori_scores.png')
st.image(scores_image)

st.write('Next, we aim to analyze the correlation between our "a-posteriori" vulnerability score and observed damage. To achieve this, we focus on a subset of 13,678 buildings located within 6 km of the five major epicenters. We compare our continuous a-posteriori vulnerability score with the a-priori one, which categorizes buildings into five levels of vulnerability, scaled from the maximum level A (highest vulnerability) to the minimum level D2 (lowest vulnerability). This analysis is presented through separated graphical representations, due to the different nature (respectively continuous and categorized) of the two vulnerability scores. In the bar chart distributions, the frequency of buildings for each damage level is plotted against both our derived a-posteriori vulnerability score (left) and the a-priori vulnerability one (right), which can be considered as a benchmark. The a-posteriori vulnerability score typically exhibits a continuous distribution that is more closely aligned with the actual damage levels. This alignment is especially pronounced for the extremes of the damage spectrum (Damage Levels D0 and D3), showing our method\'s enhanced capability in differentiating between the most and least vulnerable structures')

distro_image = load_image('app/scores_distribution_comparison.png')
st.image(distro_image)

st.write('In contrast to the previous section, where we focused on buildings within a 6 km radius of any epicenter, let us now explore the impact of varying this maximum distance. By plotting the Spearman and Kendall correlation coefficients as a function of increasing distances from the epicenters, we effectively illustrate the enhanced accuracy of our vulnerability scoring system.')
corr_image = load_image('app/correlation_vs_distance.png')
st.image(corr_image)

st.write('Looking at the figure, both the a-priori scores and the derived a-posteriori ones exhibit a predictable correlation decrease with the increases in distance from the epicenter, aligning with the expected lower impact of the earthquake. However, our a-posteriori vulnerability score consistently maintains a notably higher correlation with the damage level, even at large distances. This trend not only underlines the robustness of our approach but also highlights its predictive power in assessing earthquake vulnerability across varying proximities to epicenters.')

st.title('Credits')
st.write('This app and all methodologies showcased in it were developed by Guglielmo Ferranti, a PhD student of Complex Systems at the Department of Physics and Astronomy “Ettore Majorana” in the University of Catania, Italy.')
st.write('Special credits to the Eucentre Foundation for providing the dataset used for training the models in this app.')