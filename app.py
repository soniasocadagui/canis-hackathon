################################ Importing libraries ################################

import dash
import requests

from dash import html
from dash import dcc
from dash.dependencies import Input, Output
from io import BytesIO
from zipfile import ZipFile
from collections import Counter
from wordcloud import WordCloud
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import base64

import igraph as ig

from chart_studio import plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot

# Define random state for reproducibility
random_state = 0

# Define color palette
colors_pal = ['#6bfdeb','#22eaea','#31a0d9','#143a72','#221c3f','#FFFFFF','#000000','#EEEEEE']


################################ Creating Data for the app ################################

################ df foreign_interference_canada
# GitHub URL of the zip file
github_zip_url = "https://raw.githubusercontent.com/soniasocadagui/canis-hackathon-app/main/data/foreign_interference_canada.zip"

# Download the zip file from GitHub
response = requests.get(github_zip_url)
zip_file = ZipFile(BytesIO(response.content))

# Specify the CSV file you want to read from the zip archive
csv_file_name = "foreign_interference_canada.csv"

# Check if the CSV file exists in the zip archive
if csv_file_name in zip_file.namelist():
    # Read the CSV file into a pandas DataFrame
    with zip_file.open(csv_file_name) as file:
        df_foreign_interf = pd.read_csv(file, sep=";", low_memory=False)
        
################ df_competence_fi
df_competence_fi = pd.read_csv("https://raw.githubusercontent.com/soniasocadagui/canis-hackathon-app/main/data/df_competence_fi.csv", 
                           sep=";", low_memory=False)

################ df data_plot_when
data_plot_when = pd.read_csv("https://raw.githubusercontent.com/soniasocadagui/canis-hackathon-app/main/data/data_plot_when.csv", 
                           sep=";", low_memory=False)

################ df countries
countries = pd.read_csv("https://raw.githubusercontent.com/soniasocadagui/canis-hackathon-app/main/data/countries.csv", 
                        sep=",", encoding='latin-1')

################ df data_plot_how_much
#data_plot_how_much = pd.read_csv("https://raw.githubusercontent.com/soniasocadagui/canis-hackathon-app/main/data/data_plot_how_much3.csv",
#                                 sep=";", low_memory=False)


################################ App Visualizations ################################

##### Define data to plot 'fig_donutchart_what'
data_plot_1 = df_foreign_interf[df_foreign_interf['proxy_is_foreing_interf_canada'] != "nan"][["proxy_is_foreing_interf_canada"]].value_counts()

def donutchart_what(data_plot_1):
    # Define the list of attacks
    labels_plot = list(data_plot_1.index.get_level_values('proxy_is_foreing_interf_canada'))
    # Define number of occurences of attacks
    values_plot = list(np.ravel(df_foreign_interf[df_foreign_interf['proxy_is_foreing_interf_canada'] != "nan"][["proxy_is_foreing_interf_canada"]].value_counts()))

    # Plot donut chart
    fig = go.Figure(data=[go.Pie(labels=labels_plot, values=values_plot, hole=.4)])
    fig.update_layout(
        autosize=False,
        width=500,
        height=500,
        font_size=16,
        legend=dict(title="Is Foreign Interference?",yanchor="top",y=1.35,xanchor="left",x=0.0),
        plot_bgcolor=colors_pal[7],
        paper_bgcolor=colors_pal[7]
    )
    fig.update_traces(hoverinfo='label+value+percent', textinfo='value+percent',
                      marker=dict(colors=colors_pal))

    return fig

fig_donutchart_what = donutchart_what(data_plot_1)

##### Define data to plot 'fig_barchart_what'
# Select and arrange data to plot
pt1 = pd.DataFrame(df_competence_fi[df_competence_fi['X (Twitter) handle'].notnull()]['proxy_is_foreing_interf_canada'])
pt1['platform'] = "Twitter"
pt2 = pd.DataFrame(df_competence_fi[df_competence_fi['Facebook page'].notnull()]['proxy_is_foreing_interf_canada'])
pt2['platform'] = "Facebook"
pt3 = pd.DataFrame(df_competence_fi[df_competence_fi['Instragram page'].notnull()]['proxy_is_foreing_interf_canada'])
pt3['platform'] = "Instagram"
pt4 = pd.DataFrame(df_competence_fi[df_competence_fi['Threads account'].notnull()]['proxy_is_foreing_interf_canada'])
pt4['platform'] = "Threads"
pt5 = pd.DataFrame(df_competence_fi[df_competence_fi['YouTube account'].notnull()]['proxy_is_foreing_interf_canada'])
pt5['platform'] = "YouTube"
pt6 = pd.DataFrame(df_competence_fi[df_competence_fi['TikTok account'].notnull()]['proxy_is_foreing_interf_canada'])
pt6['platform'] = "TikTok"

data_plot_2 = pd.concat([pt1,pt2,pt3,pt4,pt5,pt6])

# Get crosstable
data_plot_2 = pd.crosstab(data_plot_2['platform'], data_plot_2['proxy_is_foreing_interf_canada'])
data_plot_2 = data_plot_2.div(data_plot_2.sum(axis=1), axis=0).reset_index()

def barchart_what(data_plot_2):
    fig = go.Figure(data=[
    go.Bar(name=data_plot_2.columns[1], x=list(data_plot_2['platform']), y=list(data_plot_2.iloc[0:,1]),marker_color=colors_pal[0]),
    go.Bar(name=data_plot_2.columns[2], x=list(data_plot_2['platform']), y=list(data_plot_2.iloc[0:,2]),marker_color=colors_pal[1])
    ])

    # Change the bar mode
    fig.update_layout(barmode='stack',
        autosize=False,
        width=600,
        height=400,
        font_size=16,
        legend=dict(title="Is Foreign Interference?"),
        plot_bgcolor=colors_pal[7],
        paper_bgcolor=colors_pal[7],
        xaxis_title="Platform", yaxis_title="Rate"
    )

    return fig

fig_barchart_what = barchart_what(data_plot_2)

##### Define data to plot 'fig_barchart_who'
# Input missing values with 0 (0 followers)
data_plot_3 = pd.concat([df_competence_fi['X (Twitter) Follower #'].fillna(0),
                         df_competence_fi['Facebook Follower #'].fillna(0),
                         df_competence_fi['Instagram Follower #'].fillna(0),
                         df_competence_fi['Threads Follower #'].fillna(0),
                         df_competence_fi['YouTube Subscriber #'].fillna(0),
                         df_competence_fi['TikTok Subscriber #'].fillna(0)],axis=1)

# Define data to plot
data_plot_3 = pd.concat([df_competence_fi,data_plot_3.mean(axis=1)],axis=1).sort_values(by=[0],ascending=False)

def barchart_who(data_plot_3):
    # Define the list of attacks
    x_axis = list(data_plot_3['Name (English)'].head(10))
    # Define number of occurences of attacks
    values_plot = list(data_plot_3[0].head(10))

    # Plot barchart
    fig = go.Figure([go.Bar(x=x_axis, y=values_plot,marker_color=px.colors.sequential.ice)])

    fig.update_layout(
        autosize=False,
        width=800,
        height=500,
        font_size=16,
        # legend=dict(title="Methods",yanchor="top",y=1.0,xanchor="right",x=1),
        plot_bgcolor=colors_pal[7],
        paper_bgcolor=colors_pal[7],
        xaxis_title="State media outlets or actors", yaxis_title="Avg # Followers"
    )

    return fig

fig_barchart_who = barchart_who(data_plot_3)

##### Define data to plot 'fig_wordcloud_who'
# Guide for Named Entity Recognition (NER) https://towardsdatascience.com/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da

# concatenate text data in one string
# text = ' '.join(features_label_cap['short_description_processed'].astype(str).tolist())

# doc = nlp(text)
# print([(X.text, X.label_) for X in doc.ents])

# Extract people mentioned
# text_list = [X.text for X in doc.ents if X.label_ in ['PERSON']]
text_list_aux = list(df_foreign_interf[df_foreign_interf['proxy_is_foreing_interf_canada'] == "Yes"]['Persons'])
text_list_aux = [x for x in text_list_aux if x is not np.nan]
text_list = []
for elem in text_list_aux:
    text_list=text_list+elem.split(",")

def wordcloud_who(text_list):
    word_could_dict = Counter(text_list)
    wordcloud = WordCloud(background_color=colors_pal[7], min_font_size=8,width=800, height=500, 
                          random_state=random_state,collocations=False,colormap='winter_r').generate_from_frequencies(word_could_dict)
    wc_img = wordcloud.to_image()
    with BytesIO() as buffer:
        wc_img.save(buffer, 'png')
        img2 = base64.b64encode(buffer.getvalue()).decode()
    
    return img2

fig_wordcloud_who = wordcloud_who(text_list)

##### Define data to plot 'fig_scatterdays_when'
data_plot_4 = data_plot_when[['day_of_week_num','day_of_week_name','proxy_is_foreing_interf_canada','Name (English)']].groupby(['day_of_week_num','day_of_week_name','proxy_is_foreing_interf_canada']).count().reset_index().sort_values(by=['day_of_week_num'])

def scatterdays_when(data_plot_4):
    y1 = data_plot_4[data_plot_4['proxy_is_foreing_interf_canada'] == "Yes"]['Name (English)']
    y1 = y1/sum(y1)

    y2 = data_plot_4[data_plot_4['proxy_is_foreing_interf_canada'] == "No"]['Name (English)']
    y2 = y2/sum(y2)

    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_plot_4[data_plot_4['proxy_is_foreing_interf_canada'] == "Yes"]['day_of_week_name'], y=y1,
                        mode='lines+markers',
                        name='Yes',line = dict(color=colors_pal[2], width=4),
                             marker = dict(color=colors_pal[2], size=12)))
    fig.add_trace(go.Scatter(x=data_plot_4[data_plot_4['proxy_is_foreing_interf_canada'] == "No"]['day_of_week_name'], y=y2,
                        mode='lines+markers',
                        name='No',line = dict(color=colors_pal[3], width=4),
                             marker = dict(color=colors_pal[3], size=12)))

    fig.update_layout(
        autosize=False,
        width=700,
        height=400,
        font_size=16,
        legend=dict(title="Is Foreign Interference?",yanchor="top",y=1,xanchor="left",x=0.3),
        plot_bgcolor=colors_pal[7],
        paper_bgcolor=colors_pal[7],
        xaxis_title="Day of Week", yaxis_title="Participation"
    )

    return fig

fig_scatterdays_when = scatterdays_when(data_plot_4)

##### Define data to plot 'fig_scatterhour_when'
data_plot_5 = data_plot_when[['hour','proxy_is_foreing_interf_canada','Name (English)']].groupby(['hour','proxy_is_foreing_interf_canada']).count().reset_index().sort_values(by=['hour'])

def scatterhour_when(data_plot_5):
    y1 = data_plot_5[data_plot_5['proxy_is_foreing_interf_canada'] == "Yes"]['Name (English)']
    y1 = y1/sum(y1)

    y2 = data_plot_5[data_plot_5['proxy_is_foreing_interf_canada'] == "No"]['Name (English)']
    y2 = y2/sum(y2)

    # Create traces
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data_plot_5[data_plot_5['proxy_is_foreing_interf_canada'] == "Yes"]['hour'], y=y1,
                        mode='lines+markers',
                        name='Yes',line = dict(color=colors_pal[2], width=4),
                             marker = dict(color=colors_pal[2], size=12)))
    fig.add_trace(go.Scatter(x=data_plot_5[data_plot_5['proxy_is_foreing_interf_canada'] == "No"]['hour'], y=y2,
                        mode='lines+markers',
                        name='No',line = dict(color=colors_pal[3], width=4),
                             marker = dict(color=colors_pal[3], size=12)))

    fig.update_layout(
        autosize=False,
        width=900,
        height=500,
        font_size=16,
        legend=dict(title="Is Foreign Interference?",yanchor="top",y=1,xanchor="left",x=0.5),
        plot_bgcolor=colors_pal[7],
        paper_bgcolor=colors_pal[7],
        xaxis_title="Hour", yaxis_title="Participation"
    )

    return fig

fig_scatterhour_when = scatterhour_when(data_plot_5)

##### Define data to plot 'fig_map_where'
data_plot_aux = df_competence_fi[['Region of Focus','Name (English)']].groupby(['Region of Focus']).count().reset_index()

data_plot_aux = pd.merge(data_plot_aux,countries,how="left",left_on=['Region of Focus'],right_on=['country'])
data_plot_aux.rename(columns={'Name (English)': "Frequency"}, inplace=True)

special_regions = ['Anglosphere','la Francophonie']

dp1 = data_plot_aux[~(data_plot_aux['Region of Focus'].isin(special_regions))]

#Canada, USA, Australia
dp2 = data_plot_aux[data_plot_aux['Region of Focus'].isin([special_regions[0]])]
lat_dp2 = dp2['lat'].str.split(",").iloc[0]
lon_dp2 = dp2['long'].str.split(",").iloc[0]

dp2_c1 = dp2.copy()
dp2_c1['lat'] = lat_dp2[0]
dp2_c1['long'] = lon_dp2[0]

dp2_c2 = dp2.copy()
dp2_c2['lat'] = lat_dp2[1]
dp2_c2['long'] = lon_dp2[1]

dp2_c3 = dp2.copy()
dp2_c3['lat'] = lat_dp2[2]
dp2_c3['long'] = lon_dp2[2]

dp2 = pd.concat([dp2_c1,dp2_c2,dp2_c3])

# France, Belgium, Canada
dp3 = data_plot_aux[data_plot_aux['Region of Focus'].isin([special_regions[1]])]
lat_dp3 = dp3['lat'].str.split(",").iloc[0]
lon_dp3 = dp3['long'].str.split(",").iloc[0]

dp3_c1 = dp3.copy()
dp3_c1['lat'] = lat_dp3[0]
dp3_c1['long'] = lon_dp3[0]

dp3_c2 = dp3.copy()
dp3_c2['lat'] = lat_dp3[1]
dp3_c2['long'] = lon_dp3[1]

dp3_c3 = dp3.copy()
dp3_c3['lat'] = lat_dp3[2]
dp3_c3['long'] = lon_dp3[2]

dp3 = pd.concat([dp3_c1,dp3_c2,dp3_c3])

data_plot_6 = pd.concat([dp1,dp2,dp3]).reset_index(drop=True)

def map_where(data_plot_6, colors_pal):
    # Get desired data and format
    data_plot_6['text'] = data_plot_6['Region of Focus'] + '<br>Frequency: ' + (data_plot_6['Frequency']).astype(str)

    # Define limits for the frequency
    limits = [(0,5),(6,10),(11,50),(51,100),(101,236)]
    
    # Define colors aplette
    colors = colors_pal

    # Define scale of points
    scale = 0.1

    fig = go.Figure()

    for i in range(len(limits)):
        lim = limits[i]
        df_sub = data_plot_6[(data_plot_6['Frequency']>=lim[0]) & (data_plot_6['Frequency']<=lim[1])]
        fig.add_trace(go.Scattergeo(
            locationmode = 'USA-states',
            lon = df_sub['long'],
            lat = df_sub['lat'],
            text = df_sub['text'],
            marker = dict(
                size = df_sub['Frequency']/scale,
                color = colors[i],
                line_color='rgb(40,40,40)',
                line_width=0.5,
                sizemode = 'area'
            ),
            name = '{0} - {1}'.format(lim[0],lim[1])))

    fig.update_layout(
            title_text = 'Frequency of events<br>(Click legend to toggle traces)',
            legend=dict(title="Frequency"),
            geo = dict(
                # scope = 'usa',
                landcolor = colors_pal[7],
            ),
            autosize=False,
            width=700,
            height=500,
            plot_bgcolor=colors_pal[7],
            paper_bgcolor=colors_pal[7]
        )

    return fig

fig_map_where = map_where(data_plot_6, colors_pal)

##### Define data to plot 'fig_barchart_why'
data_plot_7 = df_competence_fi[df_competence_fi['proxy_is_foreing_interf_canada'] == 'Yes'][['Region of Focus','Name (English)']].groupby(['Region of Focus']).count().reset_index()
data_plot_7.rename(columns={'Name (English)': "Frequency"}, inplace=True)
data_plot_7 = data_plot_7.sort_values(by=['Frequency'],ascending=False)
# # Input missing values with 0 (0 followers)
# data_plot_7 = pd.concat([df_competence_fi['X (Twitter) Follower #'].fillna(0),
#                        df_competence_fi['Facebook Follower #'].fillna(0),
#                        df_competence_fi['Instagram Follower #'].fillna(0),
#                        df_competence_fi['Threads Follower #'].fillna(0),
#                        df_competence_fi['YouTube Subscriber #'].fillna(0),
#                        df_competence_fi['TikTok Subscriber #'].fillna(0)],axis=1)

# # Define data to plot
# data_plot_7 = pd.concat([df_competence_fi,data_plot.mean(axis=1)],axis=1).sort_values(by=[0],ascending=False)

def barchart_why(data_plot_7):
    # Define the list of attacks
    x_axis = list(data_plot_7['Region of Focus'].head(10))
    # Define number of occurences of attacks
    values_plot = list(data_plot_7['Frequency'].head(10))


    # Plot barchart
    fig = go.Figure([go.Bar(x=x_axis, y=values_plot,marker_color=px.colors.sequential.ice)])

    fig.update_layout(
        autosize=False,
        width=800,
        height=500,
        font_size=16,
        # legend=dict(title="Methods",yanchor="top",y=1.0,xanchor="right",x=1),
        plot_bgcolor=colors_pal[7],
        paper_bgcolor=colors_pal[7],
        xaxis_title="Region of focus", yaxis_title="Count"
    )

    return fig

fig_barchart_why = barchart_why(data_plot_7)

##### Define data to plot 'fig_XX_how'

##### Define data to plot 'fig_barchart_howmuch'
#data_plot_8 = data_plot_how_much.copy()
data_plot_8 = df_foreign_interf[(df_foreign_interf['proxy_is_foreing_interf_canada'] != "nan") & 
                                (df_foreign_interf['date_hour'].notnull())]

# Get year, month and day
data_plot_8['year_month_day'] = pd.to_datetime(df_foreign_interf['date_hour']).dt.strftime('%Y-%m-%d')

# Get crosstable
data_plot_8 = pd.crosstab(data_plot_8['year_month_day'], data_plot_8['proxy_is_foreing_interf_canada'])
data_plot_8 = data_plot_8.div(data_plot_8.sum(axis=1), axis=0).reset_index()

# Export data for fast load in APP
#data_plot.to_csv(data_path + '/data_plot_how_much.csv', index = False, sep = ";")

def barchart_howmuch(data_plot_8):
    fig = go.Figure(data=[
    go.Bar(name=data_plot_8.columns[1], x=list(data_plot_8['year_month_day']), 
                                        y=list(data_plot_8.iloc[0:,1]),marker_color=colors_pal[0]),
    go.Bar(name=data_plot_8.columns[2], x=list(data_plot_8['year_month_day']), 
                                        y=list(data_plot_8.iloc[0:,2]),marker_color=colors_pal[1])
    ])

    # Change the bar mode
    fig.update_layout(barmode='stack',
        autosize=False,
        width=600,
        height=500,
        font_size=16,
        legend=dict(title="Methods"),
        plot_bgcolor=colors_pal[7],
        paper_bgcolor=colors_pal[7],
        xaxis_title="Tweet date", yaxis_title="Participation"
    )

    return fig

fig_barchart_howmuch = barchart_howmuch(data_plot_8)

##### Define data to plot 'fig_network_similarity'
# Define features to employ
vars_use = ['X (Twitter) Follower #','Facebook Follower #','Instagram Follower #','Threads Follower #','YouTube Subscriber #','TikTok Subscriber #']
features = df_competence_fi[vars_use].fillna(0).values
features

# Define min max scaler
scaler = MinMaxScaler()

# Transform data
features = scaler.fit_transform(features)

# Define similarity matrix
similarity = 1-pairwise_distances(features, metric="cosine")

# Define threshold of similarity to create edges
threshold = 0.9

# Create adjacency matrix
adj_matrix = similarity.copy()
adj_matrix[adj_matrix >= threshold] = 1
adj_matrix[adj_matrix < threshold] = 0
np.fill_diagonal(adj_matrix, 0)

def adjacency_matrix_to_edges(adjacency_matrix):
    edges = []
    num_nodes = len(adjacency_matrix)

    for i in range(num_nodes):
        for j in range(num_nodes):
            if adjacency_matrix[i][j] == 1:
                edges.append((j, i))  # Assuming the matrix is for a directed graph

    return edges

Edges = adjacency_matrix_to_edges(adj_matrix)

def network_similarity(Edges, df_competence_fi):
    G2=ig.Graph(Edges, directed=False)
    labels= list(df_competence_fi['Name (English)'])
    group = [1] * int(df_competence_fi.shape[0])
    N=len(labels)

    layt=G2.layout('kk', dim=3)
    Xn=[layt[k][0] for k in range(N)]# x-coordinates of nodes
    Yn=[layt[k][1] for k in range(N)]# y-coordinates
    Zn=[layt[k][2] for k in range(N)]# z-coordinates
    Xe=[]
    Ye=[]
    Ze=[]
    for e in Edges:
        Xe+=[layt[e[0]][0],layt[e[1]][0], None]# x-coordinates of edge ends
        Ye+=[layt[e[0]][1],layt[e[1]][1], None]
        Ze+=[layt[e[0]][2],layt[e[1]][2], None]

    trace1=go.Scatter3d(x=Xe,
                   y=Ye,
                   z=Ze,
                   mode='lines',
                   line=dict(color='rgb(125,125,125)', width=1),
                   hoverinfo='none'
                   )

    trace2=go.Scatter3d(x=Xn,
                   y=Yn,
                   z=Zn,
                   mode='markers',
                   name='actors',
                   marker=dict(symbol='circle',
                                 size=6,
                                 color=group,
                                 colorscale='Viridis',
                                 line=dict(color='rgb(50,50,50)', width=0.5)
                                 ),
                   text=labels,
                   hoverinfo='text'
                   )

    axis=dict(showbackground=False,
              showline=False,
              zeroline=False,
              showgrid=False,
              showticklabels=False,
              title=''
              )

    layout = go.Layout(
             title="Network of coappearances of characters in Victor Hugo's novel<br> Les Miserables (3D visualization)",
             width=1000,
             height=1000,
             showlegend=False,
             scene=dict(
                 xaxis=dict(axis),
                 yaxis=dict(axis),
                 zaxis=dict(axis),
            ),
         margin=dict(
            t=100
        ),
        hovermode='closest',
        annotations=[
               dict(
               showarrow=False,
                text="Data source: <a href='http://bost.ocks.org/mike/miserables/miserables.json'>[1] miserables.json</a>",
                xref='paper',
                yref='paper',
                x=0,
                y=0.1,
                xanchor='left',
                yanchor='bottom',
                font=dict(
                size=14
                )
                )
            ],    )


    data = [trace1, trace2]
    fig = go.Figure(data=data, layout=layout)
    #iplot(fig, filename='Les-Miserables')

    return fig

fig_network_similarity = network_similarity(Edges, df_competence_fi)


################################ Creating the app ################################
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB])
server = app.server

# the style arguments for the sidebar. We use position:fixed and a fixed width
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "2rem 1rem",
    "background-color": "#e9e5cd", #ffe7a6
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa"
}

sidebar = html.Div(
    [
        #html.H2("Content", className="display-4"),
        #html.Hr(style={'borderWidth': "0.5vh", "borderColor": "#808080","opacity": "unset"}),
        html.P(
            "Let's navigate together in this incredible visualization!", className="lead",
            style={'textAlign': 'center',
                            'color': '#000000',
                            'fontSize': 16}
        ),
        dbc.Nav(
            [
                dbc.NavLink("Intro", href="/", active="exact", style={'textAlign': 'center','fontSize': 16,
                                                                      'borderWidth': '1px',
                                                                      'borderStyle': 'dashed',
                                                                      'borderRadius': '5px',
                                                                      'margin': '3px',
                                                                      'font-weight':'bold'}),
                dbc.NavLink("What", href="/page-1", active="exact", style={'textAlign': 'center','fontSize': 16,
                                                                      'borderWidth': '1px',
                                                                      'borderStyle': 'dashed',
                                                                      'borderRadius': '5px',
                                                                      'margin': '3px',
                                                                      'font-weight':'bold'}),
                dbc.NavLink("Who", href="/page-2", active="exact", style={'textAlign': 'center','fontSize': 16,
                                                                      'borderWidth': '1px',
                                                                      'borderStyle': 'dashed',
                                                                      'borderRadius': '5px',
                                                                      'margin': '3px',
                                                                      'font-weight':'bold'}),                
                dbc.NavLink("When", href="/page-3", active="exact", style={'textAlign': 'center','fontSize': 16,
                                                                      'borderWidth': '1px',
                                                                      'borderStyle': 'dashed',
                                                                      'borderRadius': '5px',
                                                                      'margin': '3px',
                                                                      'font-weight':'bold'}),
                dbc.NavLink("Where", href="/page-4", active="exact", style={'textAlign': 'center','fontSize': 16,
                                                                      'borderWidth': '1px',
                                                                      'borderStyle': 'dashed',
                                                                      'borderRadius': '5px',
                                                                      'margin': '3px',
                                                                      'font-weight':'bold'}),
                dbc.NavLink("Why", href="/page-5", active="exact", style={'textAlign': 'center','fontSize': 16,
                                                                      'borderWidth': '1px',
                                                                      'borderStyle': 'dashed',
                                                                      'borderRadius': '5px',
                                                                      'margin': '3px',
                                                                      'font-weight':'bold'}),
                dbc.NavLink("How", href="/page-6", active="exact", style={'textAlign': 'center','fontSize': 16,
                                                                      'borderWidth': '1px',
                                                                      'borderStyle': 'dashed',
                                                                      'borderRadius': '5px',
                                                                      'margin': '3px',
                                                                      'font-weight':'bold'}),
                dbc.NavLink("How much", href="/page-7", active="exact", style={'textAlign': 'center','fontSize': 16,
                                                                      'borderWidth': '1px',
                                                                      'borderStyle': 'dashed',
                                                                     'borderRadius': '5px',
                                                                      'margin': '3px',
                                                                      'font-weight':'bold'}),
                dbc.NavLink("Model", href="/page-8", active="exact", style={'textAlign': 'center','fontSize': 16,
                                                                      'borderWidth': '1px',
                                                                      'borderStyle': 'dashed',
                                                                      'borderRadius': '5px',
                                                                      'margin': '3px',
                                                                      'font-weight':'bold'})
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)

content = html.Div(id="page-content", style=CONTENT_STYLE)

app.layout = html.Div([dcc.Location(id="url"), sidebar, content])

#### Page Intro

page_intro = html.Div([
    html.Div([

        # Logo and title
        html.Div([
            html.Img(src=app.get_asset_url('1.team_name.PNG'),
            id = 'heat-image',
            style={'height': '100px',
            'width': '500px',
            'margin-bottom': '25px',
            'textAlign': 'center'})

        ], className='one-half column'),
        html.Div([
            html.Div([
                html.H1('CANIS Data Visualization and Foreign Interference', 
                        style={'margin-bottom': '0px', 'color': '#808080','fontSize': 26, 'font-weight':'bold'}),
                html.H5('', style={'margin-bottom': '0px', 'color': '#808080'})
            ])

        ], className='one-half column', id = 'title')
        

    ], id = 'header', className='row flex-display', style={'margin-bottom': '25px'}),
    html.Div([
    html.H1('Foreign Interference?'),
    html.Div([
        html.P('As an advanced economy and open democracy, Canada is a target of foreign interference.'),
        html.P('Foreign interference includes harmful activities undertaken by foreign states, or those acting on its \
                behalf, that are clandestine, deceptive, or involve a threat to any person to advance the strategic \
                objectives of those states to the detriment of Canada’s national interests.', style={'text-align': 'justify'}), 
        html.P('Foreign interference poses one of the greatest strategic threats to Canada’s national security. Examples \
                include:'),
        html.Ul([
        html.Li("Threats, harassment or intimidation by foreign states, or those acting on its behalf, against anyone in \
                 Canada, Canadian communities, or their loved ones abroad; and,"),
        html.Li("Targeting officials at all levels of government to influence public policy and decision-making in a way \
                 that is clandestine, deceptive or threatening."),
    ])
        #html.P("https://www.publicsafety.gc.ca/cnt/ntnl-scrt/frgn-ntrfrnc/index-en.aspx")
    ]),
    html.H2('Why do states undertake foreign interference?'),
    html.Ul([
        html.Li("To undermine the integrity of democratic institutions, and covertly influence the outcomes of elections;"),
        html.Li("To sway government decision-making and policies to advance their interests, to sow distrust in society and \
                 to discredit those who threaten their interests;"),
        html.Li("To limit freedom of speech on Canadian soil by intimidating those who have come to Canada;"),
        html.Li("To obtain Canadian-made knowledge and innovation to support and advance their own military or economic objectives;"),
        html.Li("To undermine the legitimacy of Canada’s representatives abroad, or the goals of our international activities; and"),
        html.Li("To insert themselves into our supply chains and critical infrastructure."),
    ]),
    html.Div([
    html.H3("References"),
    dcc.Markdown("""
    Here are some references:

    1. [Public Safety Canada](https://www.canada.ca/en/security-intelligence-service/corporate/publications/foreign-interference-and-you/foreign-interference-and-you.html1)
    2. [Canadian Security Intelligence Service](https://www.publicsafety.gc.ca/cnt/ntnl-scrt/frgn-ntrfrnc/fi-en.aspx)
    """)
    ]),
    ]),], id='mainContainer', style={'display': 'flex', 'flex-direction': 'column'})

#### Page 1 (What)

page1_what = html.Div([
    html.H1("What tweets could be considered as foreign interference? And what couldn't?"),
    html.Div([
        dcc.Markdown(
        'Using the scrapped tweets and employing the Vader **Sentiment Analysis** technique, we categorized tweets \
         with Positive or Negative sentiment. Furthermore, using **Named Entity Recognition (NER)**, we identified \
         cities and places mentioned in each tweet. Ultimately, we established a proxy to determine what **tweets** \
         are **Foreign Interference** by flagging those with a Negative sentiment and mentions of Canada (or any city \
         from there). The following pie chart shows the percentage of Foreign Interference estimated.',
        style={'text-align': 'justify'}),
    ]),
    html.Div([dcc.Graph(id = 'donut_chart_what', figure=fig_donutchart_what, config={'displayModeBar': 'hover'}),
              ], className="grid-item"),
    html.H1("What platforms to be the focus on?"),
    html.Div([
        dcc.Markdown(
        'The following plot shows the rate of Foreign Interference per platform. Given that **XX** has the \
        **highest rate**, the **focus** should be on that platform.',
        style={'text-align': 'justify'}),
    ]),
    html.Div([dcc.Graph(id = 'bar_chart_what', figure=fig_barchart_what, config={'displayModeBar': 'hover'}),
              ], className="grid-item"),
    ])

#### Page 2 (Who)

page2_who = html.Div([
    html.H1("Who is a state foreign interfering actor? Who isn’t?"),
    html.Div([
        dcc.Markdown(
        '.',
        style={'text-align': 'justify'}),
    ]),
    #html.Div([dcc.Graph(id = 'donut_chart_who', figure=fig_donutchart_who, config={'displayModeBar': 'hover'}),
    #          ], className="grid-item"),
    html.H1("Who are the most influential state media actors based on the overall social media followers?"),
    html.Div([
        dcc.Markdown(
        '.',
        style={'text-align': 'justify'}),
    ]),
    html.Div([dcc.Graph(id = 'bar_chart_who', figure=fig_barchart_who, config={'displayModeBar': 'hover'}),
              ], className="grid-item"),
    html.H1("Who are the key individuals mentioned in the tweets related to foreign interference?"),
    html.Div([
        dcc.Markdown(
        '.',
        style={'text-align': 'justify'}),
    ]),
    html.Div([html.Img(src="data:image/png;base64," + fig_wordcloud_who),
              ], className="grid-item"),
    ])

#### Page 3 (When)

page3_when = html.Div([
    html.H1("When are the peak days for tweeting about foreign interference?"),
    html.Div([
        dcc.Markdown(
        '.',
        style={'text-align': 'justify'}),
    ]),
    html.Div([dcc.Graph(id = 'plot_scatterdays_when', figure=fig_scatterdays_when, config={'displayModeBar': 'hover'}),
              ], className="grid-item"),
    html.H1("When are the peak hours for tweeting about foreign interference?"),
    html.Div([
        dcc.Markdown(
        '.',
        style={'text-align': 'justify'}),
    ]),
    html.Div([dcc.Graph(id = 'plot_scatterhour_when', figure=fig_scatterhour_when, config={'displayModeBar': 'hover'}),
              ], className="grid-item"),
    ])

#### Page 4 (Where)

page4_where = html.Div([
    html.H1("Where are the most frequent geographic region focuses of the State media outlets or actors?"),
    html.Div([
        dcc.Markdown(
        '.',
        style={'text-align': 'justify'}),
    ]),
    html.Div([dcc.Graph(id = 'plot_map_where', figure=fig_map_where, config={'displayModeBar': 'hover'}),
              ], className="grid-item"),
    ])

#### Page 5 (Why)

page5_why = html.Div([
    html.H1("Why do certain regions become the focus of foreign interference activities?"),
    html.Div([
        dcc.Markdown(
        '.',
        style={'text-align': 'justify'}),
    ]),
    html.Div([dcc.Graph(id = 'bar_chart_why', figure=fig_barchart_why, config={'displayModeBar': 'hover'}),
              ], className="grid-item"),
    ])

#### Page 6 (How)

page6_how = html.Div([
    html.H1("How do we explain difficult concepts to senior decision-makers?"),
    html.Div([
        dcc.Markdown(
        '.',
        style={'text-align': 'justify'}),
    ]),
    #html.Div([dcc.Graph(id = 'bar_chart_why', figure=fig_barchart_why, config={'displayModeBar': 'hover'}),
    #          ], className="grid-item"),
    ])

#### Page 7 (How much)

page7_howmuch = html.Div([
    html.H1("How much has the distribution of the foreign interference changed over time?"),
    html.Div([
        dcc.Markdown(
        '.',
        style={'text-align': 'justify'}),
    ]),
    html.Div([dcc.Graph(id = 'bar_chart_howmuch', figure=fig_barchart_howmuch, config={'displayModeBar': 'hover'}),
              ], className="grid-item"),
    ])

#### Page 8 (Model)

page8_model = html.Div([
    html.H1("Graph Similarity Network"),
    html.Div([
        dcc.Markdown(
        '.',
        style={'text-align': 'justify'}),
    ]),
    html.Div([dcc.Graph(id = 'graph_network_similarity', figure=fig_network_similarity, config={'displayModeBar': 'hover'}),
              ], className="grid-item"),
    ])

@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/":
        #return html.P("This is the content of the home page!")
        return page_intro
    elif pathname == "/page-1":
        return page1_what
    elif pathname == "/page-2":
        return page2_who
    elif pathname == "/page-3":
        return page3_when
    elif pathname == "/page-4":
        return page4_where
    elif pathname == "/page-5":
        return page5_why
    elif pathname == "/page-6":
        return page6_how
    elif pathname == "/page-7":
        return page7_howmuch
    elif pathname == "/page-8":
        return page8_model
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )

if __name__ == '__main__':
    app.run(debug=False, port=(os.getenv("PORT", "1010")))
