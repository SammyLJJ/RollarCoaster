"""
Name:       Jiajian Liu
CS230:      Section 4
Data:       File RollerCoasters-Geo.csv
URL:        Link to your web application on Streamlit Cloud (if posted)

Description:
Use a python package or module we did not use in class
•	plotly
•	numpy

And this program has:
Python Features:
•	A function with two or more parameters, one of which has a default value
def find_category(column_name,info_list = info): in Location_Filter.py

•	A function that returns more than one value
def separate_empty_rows_by_column(df, column): in Data_Filter.py

•	A function that you call at least two different places in your program
def separate_empty_rows_by_column(df, column): in Data_Filter.py
def find_category(column_name,info_list = info): in Location_Filter.py

•	A list comprehension
categories = [j[index_num] for j in info[1:] if j[index_num] not in categories] in Location_Filter.py

•	A loop that iterates through items in a list, dictionary, or data frame
    for i in range(len(info_list[0])): in Location_Filter.py

        if info_list[0][i] == column_name:
            index_num = i

        for i in dict.keys():
        for j in range(len(info[0])):
            if info[0][j] == i:
                indexes[i] = j

    for i in range(len(info_list[0])): in Location_Filter.py
        if info_list[0][i] == require_column:
            index_require = i

•	At least two different methods of lists, dictionaries, or tuples.
used dictionaries and lists in Location_Filter.py

Streamlit Features:
•	At least three Streamlit different widgets  (sliders, drop downs, multi-selects, text box, etc)
double-sided sliders,download button,checkbox,multi-selects,selectbox,color select,etc.

•	Page design features (sidebar, fonts, colors, images, navigation)
multi-pages,image(icon),background-color,font color

Visualizations:
•	At least three different charts with titles, colors, labels, legends, as appropriate
Pie Chart,Bar Chart,Line Chart in Home Page

•	At least one detailed map (st.map will only get you partial credit) – for full credit, include dots, icons, text that appears when hovering over a marker, or other map features
Map included
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import csv
import matplotlib.pyplot as plt
import plotly.express as px
import pydeck as pdk

st.markdown('<h1 style="color: lightblue; text-align: center;">Find RollarCoaster in the U.S!</h1>', unsafe_allow_html=True)
st.markdown('<p style="color: lightgray; text-align: justify;">'
            ' In this page, you can choose columns to get information you want and download it by clicking "Press to Download".'
            ' Also, you can choose any column to create bar chart or pie chart which the number of each category under that column.'
            ' You can change the color of bar chart and download the chart by clicking that camera button.'
            ' There are more features such as holding your mouse on one category and the chart will show more information.'
            ' Have fun!</p>', unsafe_allow_html=True)

file_path = '/Users/sammy/Desktop/2023-1 Spring/CS 230/Project/Data/RollerCoasters-Geo.csv'

# Read the CSV file and convert it to a DataFrame
df = pd.read_csv(file_path)

column_names = df.columns
selected_info = list(st.multiselect('Select information you want:', column_names))
df1 = df[selected_info]
st.write(df1)

def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

csv = convert_df(df1)

# streamlit button widget
st.download_button(
   "Press to Download",
   csv,
   "file.csv",
   "text/csv",
   key='download-csv'
)


# Display the instruction message and the DataFrame
st.write("Select a row to show its location on the map:")
selected_index = st.selectbox("Select index:", df.index)

# Get latitude and longitude from the selected row
lat = df.loc[selected_index, 'Latitude']
lon = df.loc[selected_index, 'Longitude']

# Create a map with a marker at the selected location
location = [lat, lon]
map_data = pd.DataFrame({'lat': [lat], 'lon': [lon]})
# Specify the columns to be displayed in the tooltip
columns_to_display = ['Coaster', 'Park', 'City', 'State', 'Type', 'Design', 'Year_Opened']

# Create a tooltip string with only the specified columns
tooltip_data = df.loc[[selected_index], columns_to_display]
map_data = map_data.join(tooltip_data.reset_index(drop=True))

RED_FLAG_ICON = "https://upload.wikimedia.org/wikipedia/commons/c/c5/Red_flag_waving.svg"
#Icon or picture finder: https://commons.wikimedia.org/

icon_data = {
    "url": RED_FLAG_ICON,
    "width": 100,
    "height": 100,
    "anchorY": 1
}

map_data["icon_data"] = None
for i in map_data.index:
    map_data.at[i, "icon_data"] = icon_data

icon_layer = pdk.Layer(
    type="IconLayer",
    data=map_data,
    get_icon="icon_data",
    get_position="[lon, lat]",
    get_size=4,
    size_scale=10,
    pickable=True
)

view_state = pdk.ViewState(
    latitude=location[0],
    longitude=location[1],
    zoom=11,
    pitch=50
)

tooltip_str = '<br>'.join([f'{col}: {{{col}}}' for col in columns_to_display])


tooltip = {
    "html": f"<b>Details:</b><br> {tooltip_str}",
    "style": {
        "backgroundColor": "steelblue",
        "color": "white",
        "fontSize": "14px",
        "padding": "5px"
    }
}

icon_map = pdk.Deck(
    map_style='mapbox://styles/mapbox/light-v9',
# Go to https://docs.mapbox.com/api/maps/styles/ for more map styles
    layers=[icon_layer],
    initial_view_state=view_state,
    tooltip=tooltip
)

st.pydeck_chart(icon_map)



st.write('')
st.write('')
st.write('')
st.write('')
st.write('')
st.write('')

bar_count_column = st.selectbox('Select the column to show a chart that count number of each category:', ['Age_Group', 'Coaster', 'Park', 'City', 'State', 'Type', 'Design', 'Year_Opened', 'Top_Speed', 'Max_Height', 'Drop', 'Length', 'Duration', 'Inversions', 'Num_of_Inversions'])

chart_type = st.selectbox('Select the type of chart:', ['Bar Chart', 'Pie Chart'])

# Count the occurrences of each unique value in the selected column
value_counts = df[bar_count_column].value_counts().reset_index()

# Rename the columns to more meaningful names
value_counts.columns = [bar_count_column, 'Count']


if chart_type == 'Bar Chart':
    bar_color = st.color_picker('Select the color for the bar chart:', '#007bff')
    # Create a bar chart using plotly
    fig = px.bar(value_counts, x=bar_count_column, y='Count', text='Count', title=f'Bar Chart of Value Counts for {bar_count_column}')

    # Source: https://plotly.com/python/hover-text-and-formatting/
    fig.update_traces(hovertemplate=f'{bar_count_column}: %{{x}}<br>Count: %{{y}}', marker_color=bar_color)

    if bar_count_column == 'Year_Opened':
        show_trend = st.checkbox('Show trend line for Year_Opened data')
        if show_trend:
            trend_color = st.color_picker('Select the color for the trend:', '#007bff')
            # Calculate the trend line, source: https://plotly.com/python/hover-text-and-formatting/
            trend_line = value_counts.sort_values(by=bar_count_column).rolling(window=2).mean()

            # Add the trend line to the bar chart
            fig.add_trace(go.Scatter(x=trend_line[bar_count_column], y=trend_line['Count'], name='Trend Line',
                                     line=dict(color=trend_color)))

elif chart_type == 'Pie Chart':
    # Create a pie chart using plotly
    fig = px.pie(value_counts, names=bar_count_column, values='Count', title=f'Pie Chart of Value Counts for {bar_count_column}')

    # Source: https://plotly.com/python/hover-text-and-formatting/
    fig.update_traces(hovertemplate=f'{bar_count_column}: %{{label}}<br>Count: %{{value}}')

# Show the plot in Streamlit
st.plotly_chart(fig)



