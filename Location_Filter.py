
import streamlit as st
import pandas as pd
import csv
import plotly.express as px


st.markdown('<h1 style="color: lightblue; text-align: center;">Find RollarCoaster in the U.S!</h1>', unsafe_allow_html=True)
st.markdown('<p style="color: lightgray; text-align: justify;">Similar to data search page. '
            'You can search the park and rollar-coaster you want using the location including state, city, park '
            'and you can choose information you want and download it. </p>', unsafe_allow_html=True)

r_file = open('RollerCoasters-Geo.csv', 'r')
info = list(csv.reader(r_file))
r_file.close()

def find_category(column_name,info_list = info):

    categories = []
    for i in range(len(info_list[0])):

        if info_list[0][i] == column_name:
            index_num = i

    #use of list comprehension
    categories = [j[index_num] for j in info[1:] if j[index_num] not in categories]
    categories = list(set(categories))

    return sorted(categories)



def find_related(dict,require_column,info_list=info):
    indexes = {}
    for i in dict.keys():
        for j in range(len(info[0])):
            if info[0][j] == i:
                indexes[i] = j

    for i in range(len(info_list[0])):
        if info_list[0][i] == require_column:
            index_require = i

    require_list = []
    for i in info_list[1:]:
        for j in indexes.keys():
            if i[indexes[j]] in dict[j]:
                decision = True
            else:
                decision = False
                break

        if decision == True:
            if i[index_require] not in require_list:
                require_list.append(i[index_require])

    return sorted(require_list)

dict_select = {}

State = find_category('State')
selected_state_multi = st.multiselect('Please select state',State)
st.write('State you select is/are', selected_state_multi)


if selected_state_multi != []:
    dict_select['State'] = selected_state_multi
    City = find_related(dict_select,"City")

    # streamlit multi-selects widget
    selected_city_multi = st.multiselect('Please select city', City)
    st.write('City you select is/are', selected_city_multi)

    if selected_city_multi != []:
        dict_select['City'] = selected_city_multi
        Park = find_related(dict_select, 'Park')
        selected_park_multi = st.multiselect('Please select park', Park)
        st.write('Park you select is/are', selected_park_multi)

    else:
        Park = find_related(dict_select, 'Park')
        selected_park_multi = st.multiselect('Please select park', Park)
        st.write('Park you select is/are', selected_park_multi)

else:
    City = find_category('City')
    selected_city_multi = st.multiselect('Please select city', City)
    st.write('City you select is/are', selected_city_multi)

    if selected_city_multi != []:
        dict_select['City'] = selected_city_multi
        Park = find_related(dict_select,'Park')
        selected_park_multi = st.multiselect('Please select park', Park)
        st.write('Park you select is/are',selected_park_multi)

    else:
        Park = find_category('Park')
        selected_park_multi = st.multiselect('Please select park', Park)
        st.write('Park you select is/are', selected_park_multi)

list_info = info[0]
selected_info = st.multiselect('Select information you want:', list_info)
df = pd.DataFrame(info)
df.columns=df.iloc[0]
df1 = df[df.Park.isin(selected_park_multi)][selected_info]
st.write(df1)

# Source：https://docs.streamlit.io/knowledge-base/using-streamlit/how-download-pandas-dataframe-csv
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

st.markdown('<p style="color: lightgray; text-align: justify;">The Map will show information of all parks '
            'in the U.S if you didn\'t choose any park using the "Please select park" selectbox above. '
            'When you hover over a point, it will show only longitude and latitude of this park. '
            'If you want to let it show other information, in "Select the information you want" box, '
            'click the information you want to show on the map!</p>', unsafe_allow_html=True)

data = pd.read_csv("/Users/sammy/Desktop/2023-1 Spring/CS 230/Project/Data/RollerCoasters-Geo.csv")
if selected_park_multi == []:
    data_selected = data.copy()
else:
    data_selected = data[data.Park.isin(selected_park_multi)]
print(data_selected)

plot_color = st.color_picker('Select the color for the dot:', '#007bff')
fig = px.scatter_mapbox(
    data_selected,
    lat="Latitude",
    lon="Longitude",
    hover_data=selected_info,  # Modify this list with the columns you want to display on hover
    color_discrete_sequence=[plot_color],
    zoom=3,
    height=600,
)

# Set the Mapbox access token (replace with your own token)
fig.update_layout(mapbox_style="open-street-map")
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})

# Show the plot in Streamlit
st.plotly_chart(fig)
