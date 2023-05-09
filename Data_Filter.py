import streamlit as st
import pandas as pd
import csv
import numpy as np





st.markdown('<h1 style="color: lightblue; text-align: center;">Find RollarCoaster in the U.S!</h1>', unsafe_allow_html=True)


st.markdown('<p style="color: lightgray; text-align: justify;">You can use this page to filter roller coaster information in a certain range.'
            'It should be noted that some data in our database has not been collected.'
            'If you do not check "keep the empty data" button, it will automatically delete rows with blank data.'
            'However，you can open database "Data_Ignored" to see what rows are deleted.'
            'Also, all database you see could be download as a csv file by clicking "Press to Download" button.'
            ' Have fun!</p>', unsafe_allow_html=True)

pd.set_option('display.max_columns', None)  # Display all columns
pd.set_option('display.max_rows', None)     # Display all rows
pd.set_option('display.width', None)        # Auto-detect the display width
pd.set_option('display.max_colwidth', None) # Display the full width of each column




file_path = '/Users/sammy/Desktop/2023-1 Spring/CS 230/Project/Data/RollerCoasters-Geo.csv'

# Read the CSV file and convert it to a DataFrame
df = pd.read_csv(file_path)

def separate_empty_rows_by_column(df, column):
    # Check if the column exists in the DataFrame
    if column not in df.columns:
        raise ValueError(f"The specified column '{column}' does not exist in the DataFrame.")

    # Create a boolean mask to check if any value is empty (empty string or None) in the specified column
    mask = (df[column] == '') | df[column].isnull()

    # Separate the rows with empty values into a new DataFrame
    df_with_empty_values = df[mask]

    # Remove the rows with empty values from the original DataFrame
    cleaned_df = df[~mask]

    return cleaned_df, df_with_empty_values


def filter_rows_by_range(df, column, value_range):
    # Check if the column exists in the DataFrame
    if column not in df.columns:
        raise ValueError(f"The specified column '{column}' does not exist in the DataFrame.")

    # Ensure the input range is a tuple with exactly two elements
    if not isinstance(value_range, tuple) or len(value_range) != 2:
        raise ValueError("The 'value_range' argument must be a tuple with exactly two elements.")

    # Filter rows based on the specified column and range
    filtered_df = df[(df[column] >= value_range[0]) & (df[column] <= value_range[1])]

    return filtered_df

def return_result(df,column_range,value_range,column,check = True):
    if check == True:
        st.write("User chose to keep empty values.")
        st.write('')
        filter = filter_rows_by_range(df,column_range,value_range)
        values = filter[column].astype(float)
        min_value = min(values) if not np.isnan(min(values)) else 0
        max_value = max(values) if not np.isnan(max(values)) else 250

        return min_value,max_value

    elif check == False:
        st.write("User chose not to keep empty values.")
        st.write('')
        filter = filter_rows_by_range(df, column_range, value_range)
        cleaned_df, df_with_empty_values = separate_empty_rows_by_column(filter, column)
        values = cleaned_df[column].astype(float)

        min_value = min(values) if not np.isnan(min(values)) else 0
        max_value = max(values) if not np.isnan(max(values)) else 250

        return min_value,max_value,cleaned_df, df_with_empty_values


def data_to_see(widget_key,download_button_key,download_button_key2,data_left,data_ignored = pd.DataFrame()):

    dataframes = {
        'Data_Left': data_left,
        'Data_Ignored': data_ignored
    }
    selections = st.multiselect(
        'Select the DataFrames you want to see:',
        options=list(dataframes.keys()),
        key=widget_key
    )

    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    # Display the selected DataFrames
    if selections == ['Data_Left']:
        st.write('Data_Left')
        st.write(dataframes['Data_Left'])
        csv = convert_df(dataframes['Data_Left'])

        # streamlit button widget

        st.download_button(
            "Press to Download",
            csv,
            "file.csv",
            "text/csv",
            key=download_button_key
        )
    elif selections == ['Data_Ignored']:
        st.write('Data_Ignored')
        st.write(dataframes['Data_Ignored'])
        csv = convert_df(dataframes['Data_Ignored'])

        # streamlit button widget

        st.download_button(
            "Press to Download",
            csv,
            "file.csv",
            "text/csv",
            key=download_button_key2
        )
    elif selections == []:
        st.write('You didn\'t choose any dataframe')
    else:
        st.write(f"{selections[0]}:")
        st.write(dataframes[selections[0]])
        csv = convert_df(dataframes[selections[0]])

        # streamlit button widget

        st.download_button(
            "Press to Download",
            csv,
            "file.csv",
            "text/csv",
            key=download_button_key
        )

        st.write(f"{selections[1]}:")
        st.write(dataframes[selections[1]])
        csv = convert_df(dataframes[selections[1]])

        # streamlit button widget

        st.download_button(
            "Press to Download",
            csv,
            "file.csv",
            "text/csv",
            key=download_button_key2
        )


st.write('')
st.write('')
st.write('')
st.write('')
keep_empty_values_speed = st.checkbox("Keep rows with empty values of speed?")
if keep_empty_values_speed: #Speed
    st.write("User chose to keep empty values.")
    st.write('')
    speed_values = df['Top_Speed'].astype(float)
    min_speed = min(speed_values)
    max_speed = max(speed_values)
    speed_range = st.slider('Select the range of Top_Speed', min_value=min_speed, max_value=max_speed,
                            value=(min_speed, max_speed))

    data_to_see('df_1','download_1','2download_1',df)


    st.write('')
    st.write('')
    st.write('')
    keep_empty_values_height = st.checkbox("Keep rows with empty values of height?")
    if keep_empty_values_height: #Height
        min_height,max_height = return_result(df, 'Top_Speed', speed_range, 'Max_Height', check=True)
        height_range = st.slider('Select the range of Max_Height', min_value=min_height, max_value=max_height,
                                value=(min_height, max_height))
        data_to_see('df_1_1','download_1_1','2download_1_1',df)

        st.write('')
        st.write('')
        st.write('')
        keep_empty_values_drop = st.checkbox("Keep rows with empty values of drop?")
        if keep_empty_values_drop:  # Drop
            min_drop, max_drop = return_result(df, 'Max_Height', height_range, 'Drop', check=True)
            drop_range = st.slider('Select the range of Drop', min_value=min_drop, max_value=max_drop,
                                     value=(min_drop, max_drop))

            data_to_see('df_1_1_1','download_1_1_1','2download_1_1_1',df)

            st.write('')
            st.write('')
            st.write('')
            keep_empty_values_length = st.checkbox("Keep rows with empty values of Length?")
            if keep_empty_values_length:  # Length
                min_Length, max_Length = return_result(df, 'Drop', drop_range, 'Length', check=True)
                Length_range = st.slider('Select the range of Length', min_value=min_Length, max_value=max_Length,
                                         value=(min_Length, max_Length))

                data_to_see('df_1_1_1_1', 'download_1_1_1_1', '2download_1_1_1_1', df)

                st.write('')
                st.write('')
                st.write('')
                keep_empty_values_Duration = st.checkbox("Keep rows with empty values of Duration?")
                if keep_empty_values_Duration:  # Duration
                    min_Duration, max_Duration = return_result(df, 'Length', Length_range, 'Duration', check=True)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_1_1_1_1_1', 'download_1_1_1_1_1', '2download_1_1_1_1_1', df)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(df, 'Duration', Duration_range))

                else:  # Duration
                    min_Duration, max_Duration, cleaned_df_Duration, df_with_empty_values_Duration = return_result(df,
                                                                                                                   'Length',
                                                                                                                   Length_range,
                                                                                                                   'Duration',
                                                                                                                   check=False)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_1_1_1_1_2', 'download_1_1_1_1_2', '2download_1_1_1_1_2', cleaned_df_Duration,
                                df_with_empty_values_Duration)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_Duration, 'Duration', Duration_range))

            else:  # Length
                min_Length, max_Length, cleaned_df_Length, df_with_empty_values_Length = return_result(df, 'Drop',
                                                                                                       drop_range,
                                                                                                       'Length',
                                                                                                       check=False)
                Length_range = st.slider('Select the range of Length', min_value=min_Length, max_value=max_Length,
                                         value=(min_Length, max_Length))

                data_to_see('df_1_1_1_2', 'download_1_1_1_2', '2download_1_1_1_2', cleaned_df_Length,
                            df_with_empty_values_Length)

                st.write('')
                st.write('')
                st.write('')
                keep_empty_values_Duration = st.checkbox("Keep rows with empty values of Duration?")
                if keep_empty_values_Duration:  # Duration
                    min_Duration, max_Duration = return_result(cleaned_df_Length, 'Length', Length_range, 'Duration', check=True)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_1_1_1_2_1', 'download_1_1_1_2_1', '2download_1_1_1_2_1', cleaned_df_Length)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_Length, 'Duration', Duration_range))

                else:  # Duration
                    min_Duration, max_Duration, cleaned_df_Duration, df_with_empty_values_Duration = return_result(cleaned_df_Length,
                                                                                                                   'Length',
                                                                                                                   Length_range,
                                                                                                                   'Duration',
                                                                                                                   check=False)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_1_1_1_2_2', 'download_1_1_1_2_2', '2download_1_1_1_2_2', cleaned_df_Duration,
                                df_with_empty_values_Duration)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_Duration, 'Duration', Duration_range))



        else:  # Drop
            min_drop, max_drop, cleaned_df_drop, df_with_empty_values_drop = return_result(df, 'Max_Height', height_range, 'Drop', check=False)
            drop_range = st.slider('Select the range of Drop', min_value=min_drop, max_value=max_drop,
                                   value=(min_drop, max_drop))

            data_to_see('df_1_1_2','download_1_1_2','2download_1_1_2',cleaned_df_drop,df_with_empty_values_drop)



            st.write('')
            st.write('')
            st.write('')
            keep_empty_values_length = st.checkbox("Keep rows with empty values of Length?")
            if keep_empty_values_length:  # Length
                min_Length, max_Length = return_result(cleaned_df_drop, 'Drop', drop_range, 'Length', check=True)
                Length_range = st.slider('Select the range of Length', min_value=min_Length, max_value=max_Length,
                                         value=(min_Length, max_Length))

                data_to_see('df_1_1_2_1','download_1_1_2_1','2download_1_1_2_1', cleaned_df_drop)

                st.write('')
                st.write('')
                st.write('')
                keep_empty_values_Duration = st.checkbox("Keep rows with empty values of Duration?")
                if keep_empty_values_Duration:  # Duration
                    min_Duration, max_Duration = return_result(cleaned_df_drop, 'Length', Length_range, 'Duration', check=True)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_1_1_2_1_1','download_1_1_2_1_1','2download_1_1_2_1_1', cleaned_df_drop)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_drop, 'Duration', Duration_range))

                else:  # Duration
                    min_Duration, max_Duration, cleaned_df_Duration, df_with_empty_values_Duration = return_result(cleaned_df_drop,
                                                                                                                   'Length',
                                                                                                                   Length_range,
                                                                                                                   'Duration',
                                                                                                                   check=False)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_1_1_2_1_2','download_1_1_2_1_2','2download_1_1_2_1_2', cleaned_df_Duration,
                                df_with_empty_values_Duration)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_Duration, 'Duration', Duration_range))





            else:  # Length
                min_Length, max_Length, cleaned_df_Length, df_with_empty_values_Length = return_result(cleaned_df_drop, 'Drop',
                                                                                                       drop_range,
                                                                                                       'Length',
                                                                                                       check=False)
                Length_range = st.slider('Select the range of Length', min_value=min_Length, max_value=max_Length,
                                         value=(min_Length, max_Length))

                data_to_see('df_1_1_2_2','download_1_1_2_2','2download_1_1_2_2', cleaned_df_Length,
                            df_with_empty_values_Length)

                st.write('')
                st.write('')
                st.write('')
                keep_empty_values_Duration = st.checkbox("Keep rows with empty values of Duration?")
                if keep_empty_values_Duration:  # Duration
                    min_Duration, max_Duration = return_result(cleaned_df_Length, 'Length', Length_range, 'Duration',
                                                               check=True)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_1_1_2_2_1', 'download_1_1_2_2_1', '2download_1_1_2_2_1', cleaned_df_Length)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_Length, 'Duration', Duration_range))

                else:  # Duration
                    min_Duration, max_Duration, cleaned_df_Duration, df_with_empty_values_Duration = return_result(
                        cleaned_df_Length,
                        'Length',
                        Length_range,
                        'Duration',
                        check=False)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_1_1_2_2_2', 'download_1_1_2_2_2', '2download_1_1_2_2_2', cleaned_df_Duration,
                                df_with_empty_values_Duration)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_Duration, 'Duration', Duration_range))




    else: #Height
        min_height, max_height, cleaned_df_height, df_with_empty_values_height = return_result(df, 'Top_Speed', speed_range, 'Max_Height', check=False)
        height_range = st.slider('Select the range of Max_Height', min_value=min_height, max_value=max_height,
                                 value=(min_height, max_height))

        data_to_see('df_1_2','download_1_2','2download_1_2',cleaned_df_height,df_with_empty_values_height)

        st.write('')
        st.write('')
        st.write('')
        keep_empty_values_drop = st.checkbox("Keep rows with empty values of drop?")
        if keep_empty_values_drop:  # Drop
            min_drop, max_drop = return_result(cleaned_df_height, 'Max_Height', height_range, 'Drop', check=True)
            drop_range = st.slider('Select the range of Drop', min_value=min_drop, max_value=max_drop,
                                     value=(min_drop, max_drop))

            data_to_see('df_1_2_1','download_1_2_1','2download_1_2_1',cleaned_df_height)

            st.write('')
            st.write('')
            st.write('')
            keep_empty_values_length = st.checkbox("Keep rows with empty values of Length?")
            if keep_empty_values_length:  # Length
                min_Length, max_Length = return_result(cleaned_df_height, 'Drop', drop_range, 'Length', check=True)
                Length_range = st.slider('Select the range of Length', min_value=min_Length, max_value=max_Length,
                                         value=(min_Length, max_Length))

                data_to_see('df_1_2_1_1','download_1_2_1_1','2download_1_2_1_1', cleaned_df_height)

                st.write('')
                st.write('')
                st.write('')
                keep_empty_values_Duration = st.checkbox("Keep rows with empty values of Duration?")
                if keep_empty_values_Duration:  # Duration
                    min_Duration, max_Duration = return_result(cleaned_df_height, 'Length', Length_range, 'Duration',
                                                               check=True)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_1_2_1_1_1', 'download_1_2_1_1_1', '2download_1_2_1_1_1', cleaned_df_height)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_height, 'Duration', Duration_range))

                else:  # Duration
                    min_Duration, max_Duration, cleaned_df_Duration, df_with_empty_values_Duration = return_result(
                        cleaned_df_height, 'Length',
                        Length_range,
                        'Duration',
                        check=False)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_1_2_1_1_2', 'download_1_2_1_1_2', '2download_1_2_1_1_2', cleaned_df_Duration,
                                df_with_empty_values_Duration)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_Duration, 'Duration', Duration_range))


            else:  # Length
                min_Length, max_Length, cleaned_df_Length, df_with_empty_values_Length = return_result(cleaned_df_height, 'Drop',
                                                                                                       drop_range,
                                                                                                       'Length',
                                                                                                       check=False)
                Length_range = st.slider('Select the range of Length', min_value=min_Length, max_value=max_Length,
                                         value=(min_Length, max_Length))

                data_to_see('df_1_2_1_2','download_1_2_1_2','2download_1_2_1_2', cleaned_df_Length,
                            df_with_empty_values_Length)

                st.write('')
                st.write('')
                st.write('')
                keep_empty_values_Duration = st.checkbox("Keep rows with empty values of Duration?")
                if keep_empty_values_Duration:  # Duration
                    min_Duration, max_Duration = return_result(cleaned_df_Length, 'Length', Length_range, 'Duration',
                                                               check=True)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_1_2_1_2_1', 'download_1_2_1_2_1', '2download_1_2_1_2_1', cleaned_df_Length)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_Length, 'Duration', Duration_range))

                else:  # Duration
                    min_Duration, max_Duration, cleaned_df_Duration, df_with_empty_values_Duration = return_result(
                        cleaned_df_Length, 'Length',
                        Length_range,
                        'Duration',
                        check=False)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_1_2_1_2_2', 'download_1_2_1_2_2', '2download_1_2_1_2_2', cleaned_df_Duration,
                                df_with_empty_values_Duration)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_Duration, 'Duration', Duration_range))


        else:  # Drop
            min_drop, max_drop, cleaned_df_drop, df_with_empty_values_drop = return_result(cleaned_df_height, 'Max_Height', height_range, 'Drop', check=False)
            drop_range = st.slider('Select the range of Drop', min_value=min_drop, max_value=max_drop,
                                   value=(min_drop, max_drop))

            data_to_see('df_1_2_2','download_1_2_2','2download_1_2_2',cleaned_df_drop,df_with_empty_values_drop)

            st.write('')
            st.write('')
            st.write('')
            keep_empty_values_length = st.checkbox("Keep rows with empty values of Length?")
            if keep_empty_values_length:  # Length
                min_Length, max_Length = return_result(cleaned_df_drop, 'Drop', drop_range, 'Length', check=True)
                Length_range = st.slider('Select the range of Length', min_value=min_Length, max_value=max_Length,
                                         value=(min_Length, max_Length))

                data_to_see('df_1_2_2_1','download_1_2_2_1','2download_1_2_2_1', cleaned_df_drop)

                st.write('')
                st.write('')
                st.write('')
                keep_empty_values_Duration = st.checkbox("Keep rows with empty values of Duration?")
                if keep_empty_values_Duration:  # Duration
                    min_Duration, max_Duration = return_result(cleaned_df_drop, 'Length', Length_range, 'Duration',
                                                               check=True)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_1_2_2_1_1', 'download_1_2_2_1_1', '2download_1_2_2_1_1', cleaned_df_drop)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_drop, 'Duration', Duration_range))

                else:  # Duration
                    min_Duration, max_Duration, cleaned_df_Duration, df_with_empty_values_Duration = return_result(
                        cleaned_df_drop, 'Length',
                        Length_range,
                        'Duration',
                        check=False)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_1_2_2_1_2', 'download_1_2_2_1_2', '2download_1_2_2_1_2', cleaned_df_Duration,
                                df_with_empty_values_Duration)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_Duration, 'Duration', Duration_range))





            else:  # Length
                min_Length, max_Length, cleaned_df_Length, df_with_empty_values_Length = return_result(cleaned_df_drop, 'Drop',
                                                                                                       drop_range,
                                                                                                       'Length',
                                                                                                       check=False)
                Length_range = st.slider('Select the range of Length', min_value=min_Length, max_value=max_Length,
                                         value=(min_Length, max_Length))

                data_to_see('df_1_2_2_2','download_1_2_2_2','2download_1_2_2_2', cleaned_df_Length,
                            df_with_empty_values_Length)

                st.write('')
                st.write('')
                st.write('')
                keep_empty_values_Duration = st.checkbox("Keep rows with empty values of Duration?")
                if keep_empty_values_Duration:  # Duration
                    min_Duration, max_Duration = return_result(cleaned_df_Length, 'Length', Length_range, 'Duration',
                                                               check=True)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_1_2_2_2_1','download_1_2_2_2_1','2download_1_2_2_2_1', cleaned_df_Length)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_Length, 'Duration', Duration_range))

                else:  # Duration
                    min_Duration, max_Duration, cleaned_df_Duration, df_with_empty_values_Duration = return_result(
                        cleaned_df_Length, 'Length',
                        Length_range,
                        'Duration',
                        check=False)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_1_2_2_2_2','download_1_2_2_2_2','2download_1_2_2_2_2', cleaned_df_Duration,
                                df_with_empty_values_Duration)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_Duration, 'Duration', Duration_range))


else: #Speed
    cleaned_df_speed, df_with_empty_values_speed = separate_empty_rows_by_column(df, 'Top_Speed')
    st.write("User chose not to keep empty values.")
    st.write('')
    speed_values = cleaned_df_speed['Top_Speed'].astype(float)
    min_speed = min(speed_values)
    max_speed = max(speed_values)
    speed_range = st.slider('Select the range of Top_Speed', min_value=min_speed, max_value=max_speed,
                                 value=(min_speed, max_speed))

    data_to_see('df_2','download_2','2download_2',cleaned_df_speed,df_with_empty_values_speed)

    st.write('')
    st.write('')
    st.write('')

    keep_empty_values_height = st.checkbox("Keep rows with empty values of height?")
    if keep_empty_values_height: #Height
        min_height,max_height = return_result(cleaned_df_speed, 'Top_Speed', speed_range, 'Max_Height', check=True)
        height_range = st.slider('Select the range of Max_Height', min_value=min_height, max_value=max_height,
                                value=(min_height, max_height))
        data_to_see('df_2_1', 'download_2_1', '2download_2_1', cleaned_df_speed)

        st.write('')
        st.write('')
        st.write('')
        keep_empty_values_drop = st.checkbox("Keep rows with empty values of drop?")
        if keep_empty_values_drop:  # Drop
            min_drop, max_drop = return_result(cleaned_df_speed, 'Max_Height', height_range, 'Drop', check=True)
            drop_range = st.slider('Select the range of Drop', min_value=min_drop, max_value=max_drop,
                                     value=(min_drop, max_drop))
            data_to_see('df_2_1_1','download_2_1_1','2download_2_1_1',cleaned_df_speed)

            st.write('')
            st.write('')
            st.write('')
            keep_empty_values_length = st.checkbox("Keep rows with empty values of Length?")
            if keep_empty_values_length:  # Length
                min_Length, max_Length = return_result(cleaned_df_speed, 'Drop', drop_range, 'Length',
                                                       check=True)
                Length_range = st.slider('Select the range of Length', min_value=min_Length, max_value=max_Length,
                                         value=(min_Length, max_Length))

                data_to_see('df_2_1_1_1','download_2_1_1_1','2download_2_1_1_1', cleaned_df_speed)

                st.write('')
                st.write('')
                st.write('')
                keep_empty_values_Duration = st.checkbox("Keep rows with empty values of Duration?")
                if keep_empty_values_Duration:  # Duration
                    min_Duration, max_Duration = return_result(cleaned_df_speed, 'Length', Length_range, 'Duration',
                                                               check=True)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_2_1_1_1_1', 'download_2_1_1_1_1', '2download_2_1_1_1_1', cleaned_df_speed)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_speed, 'Duration', Duration_range))

                else:  # Duration
                    min_Duration, max_Duration, cleaned_df_Duration, df_with_empty_values_Duration = return_result(
                        cleaned_df_speed, 'Length',
                        Length_range,
                        'Duration',
                        check=False)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_2_1_1_1_2', 'download_2_1_1_1_2', '2download_2_1_1_1_2', cleaned_df_Duration,
                                df_with_empty_values_Duration)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_Duration, 'Duration', Duration_range))


            else:  # Length
                min_Length, max_Length, cleaned_df_Length, df_with_empty_values_Length = return_result(cleaned_df_speed,
                                                                                                       'Drop',
                                                                                                       drop_range,
                                                                                                       'Length',
                                                                                                       check=False)
                Length_range = st.slider('Select the range of Length', min_value=min_Length, max_value=max_Length,
                                         value=(min_Length, max_Length))

                data_to_see('df_2_1_1_2','download_2_1_1_2','2download_2_1_1_2', cleaned_df_Length,
                            df_with_empty_values_Length)

                st.write('')
                st.write('')
                st.write('')
                keep_empty_values_Duration = st.checkbox("Keep rows with empty values of Duration?")
                if keep_empty_values_Duration:  # Duration
                    min_Duration, max_Duration = return_result(cleaned_df_Length, 'Length', Length_range, 'Duration',
                                                               check=True)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_2_1_1_2_1','download_2_1_1_2_1','2download_2_1_1_2_1', cleaned_df_Length)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_Length, 'Duration', Duration_range))

                else:  # Duration
                    min_Duration, max_Duration, cleaned_df_Duration, df_with_empty_values_Duration = return_result(
                        cleaned_df_Length, 'Length',
                        Length_range,
                        'Duration',
                        check=False)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_1_1_1_1_2_2', 'download_1_1_1_1_2_2', '2download_1_1_1_1_2_2', cleaned_df_Duration,
                                df_with_empty_values_Duration)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    st.write(filter_rows_by_range(cleaned_df_Duration, 'Duration', Duration_range))

        else:  # Drop
            min_drop, max_drop, cleaned_df_drop, df_with_empty_values_drop = return_result(cleaned_df_speed, 'Max_Height', height_range, 'Drop', check=False)
            drop_range = st.slider('Select the range of Drop', min_value=min_drop, max_value=max_drop,
                                   value=(min_drop, max_drop))

            data_to_see('df_2_1_2','download_2_1_2','2download_2_1_2',cleaned_df_drop, df_with_empty_values_drop)

            st.write('')
            st.write('')
            st.write('')
            keep_empty_values_length = st.checkbox("Keep rows with empty values of Length?")
            if keep_empty_values_length:  # Length
                min_Length, max_Length = return_result(cleaned_df_drop, 'Drop', drop_range, 'Length',
                                                       check=True)
                Length_range = st.slider('Select the range of Length', min_value=min_Length, max_value=max_Length,
                                         value=(min_Length, max_Length))

                data_to_see('df_2_1_2_1','download_2_1_2_1','2download_2_1_2_1', cleaned_df_drop)

                st.write('')
                st.write('')
                st.write('')
                keep_empty_values_Duration = st.checkbox("Keep rows with empty values of Duration?")
                if keep_empty_values_Duration:  # Duration
                    min_Duration, max_Duration = return_result(cleaned_df_drop, 'Length', Length_range, 'Duration',
                                                               check=True)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_2_1_2_1_1','download_2_1_2_1_1','2download_2_1_2_1_1', cleaned_df_drop)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_drop, 'Duration', Duration_range))

                else:  # Duration
                    min_Duration, max_Duration, cleaned_df_Duration, df_with_empty_values_Duration = return_result(
                        cleaned_df_drop, 'Length',
                        Length_range,
                        'Duration',
                        check=False)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_2_1_2_1_2','download_2_1_2_1_2','2download_2_1_2_1_2', cleaned_df_Duration,
                                df_with_empty_values_Duration)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_Duration, 'Duration', Duration_range))


            else:  # Length
                min_Length, max_Length, cleaned_df_Length, df_with_empty_values_Length = return_result(cleaned_df_drop,
                                                                                                       'Drop',
                                                                                                       drop_range,
                                                                                                       'Length',
                                                                                                       check=False)
                Length_range = st.slider('Select the range of Length', min_value=min_Length, max_value=max_Length,
                                         value=(min_Length, max_Length))

                data_to_see('df_2_1_2_2','download_2_1_2_2','2download_2_1_2_2', cleaned_df_Length,
                            df_with_empty_values_Length)

                st.write('')
                st.write('')
                st.write('')
                keep_empty_values_Duration = st.checkbox("Keep rows with empty values of Duration?")
                if keep_empty_values_Duration:  # Duration
                    min_Duration, max_Duration = return_result(cleaned_df_Length, 'Length', Length_range, 'Duration',
                                                               check=True)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_2_1_2_2_1', 'download_2_1_2_2_1', '2download_2_1_2_2_1', cleaned_df_Length)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_Length, 'Duration', Duration_range))

                else:  # Duration
                    min_Duration, max_Duration, cleaned_df_Duration, df_with_empty_values_Duration = return_result(
                        cleaned_df_Length, 'Length',
                        Length_range,
                        'Duration',
                        check=False)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_2_1_2_2_2', 'download_2_1_2_2_2', '2download_2_1_2_2_2', cleaned_df_Duration,
                                df_with_empty_values_Duration)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_Duration, 'Duration', Duration_range))


    else: #Height
        min_height, max_height, cleaned_df_height, df_with_empty_values_height = return_result(cleaned_df_speed, 'Top_Speed', speed_range, 'Max_Height', check=False)
        height_range = st.slider('Select the range of Max_Height', min_value=min_height, max_value=max_height,
                                 value=(min_height, max_height))

        data_to_see('df_2_2','download_2_2','2download_2_2',cleaned_df_height, df_with_empty_values_height)

        st.write('')
        st.write('')
        st.write('')
        keep_empty_values_drop = st.checkbox("Keep rows with empty values of drop?")
        if keep_empty_values_drop:  # Drop
            min_drop, max_drop = return_result(cleaned_df_height, 'Max_Height', height_range, 'Drop', check=True)
            drop_range = st.slider('Select the range of Drop', min_value=min_drop, max_value=max_drop,
                                     value=(min_drop, max_drop))

            data_to_see('df_2_2_1','download_2_2_1','2download_2_2_1',cleaned_df_height)

            st.write('')
            st.write('')
            st.write('')
            keep_empty_values_length = st.checkbox("Keep rows with empty values of Length?")
            if keep_empty_values_length:  # Length
                min_Length, max_Length = return_result(cleaned_df_height, 'Drop', drop_range, 'Length',
                                                       check=True)
                Length_range = st.slider('Select the range of Length', min_value=min_Length, max_value=max_Length,
                                         value=(min_Length, max_Length))

                data_to_see('df_2_2_1_1','download_2_2_1_1','2download_2_2_1_1', cleaned_df_height)

                st.write('')
                st.write('')
                st.write('')
                keep_empty_values_Duration = st.checkbox("Keep rows with empty values of Duration?")
                if keep_empty_values_Duration:  # Duration
                    min_Duration, max_Duration = return_result(cleaned_df_height, 'Length', Length_range, 'Duration',
                                                               check=True)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_2_2_1_1_1','download_2_2_1_1_1','2download_2_2_1_1_1', cleaned_df_height)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_height, 'Duration', Duration_range))

                else:  # Duration
                    min_Duration, max_Duration, cleaned_df_Duration, df_with_empty_values_Duration = return_result(
                        cleaned_df_height, 'Length',
                        Length_range,
                        'Duration',
                        check=False)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_2_2_1_1_2','download_2_2_1_1_2','2download_2_2_1_1_2', cleaned_df_Duration,
                                df_with_empty_values_Duration)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_Duration, 'Duration', Duration_range))





            else:  # Length
                min_Length, max_Length, cleaned_df_Length, df_with_empty_values_Length = return_result(cleaned_df_height,
                                                                                                       'Drop',
                                                                                                       drop_range,
                                                                                                       'Length',
                                                                                                       check=False)
                Length_range = st.slider('Select the range of Length', min_value=min_Length, max_value=max_Length,
                                         value=(min_Length, max_Length))

                data_to_see('df_2_2_1_2','download_2_2_1_2','2download_2_2_1_2', cleaned_df_Length,
                            df_with_empty_values_Length)

                st.write('')
                st.write('')
                st.write('')
                keep_empty_values_Duration = st.checkbox("Keep rows with empty values of Duration?")
                if keep_empty_values_Duration:  # Duration
                    min_Duration, max_Duration = return_result(cleaned_df_Length, 'Length', Length_range, 'Duration',
                                                               check=True)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_2_2_1_2_1','download_2_2_1_2_1','2download_2_2_1_2_1', cleaned_df_Length)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_Length, 'Duration', Duration_range))

                else:  # Duration
                    min_Duration, max_Duration, cleaned_df_Duration, df_with_empty_values_Duration = return_result(
                        cleaned_df_Length, 'Length',
                        Length_range,
                        'Duration',
                        check=False)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_1_1_1_1_2_2', 'download_1_1_1_1_2_2', '2download_1_1_1_1_2_2', cleaned_df_Duration,
                                df_with_empty_values_Duration)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_Duration, 'Duration', Duration_range))





        else:  # Drop
            min_drop, max_drop, cleaned_df_drop, df_with_empty_values_drop = return_result(cleaned_df_height, 'Max_Height', height_range, 'Drop', check=False)
            drop_range = st.slider('Select the range of Drop', min_value=min_drop, max_value=max_drop,
                                   value=(min_drop, max_drop))
            data_to_see('df_2_2_2','download_2_2_2','2download_2_2_2',cleaned_df_drop, df_with_empty_values_drop)

            st.write('')
            st.write('')
            st.write('')
            keep_empty_values_length = st.checkbox("Keep rows with empty values of Length?")
            if keep_empty_values_length:  # Length
                min_Length, max_Length = return_result(cleaned_df_drop, 'Drop', drop_range, 'Length',
                                                       check=True)
                Length_range = st.slider('Select the range of Length', min_value=min_Length, max_value=max_Length,
                                         value=(min_Length, max_Length))

                data_to_see('df_2_2_2_1','download_2_2_2_1','2download_2_2_2_1', cleaned_df_drop)

                st.write('')
                st.write('')
                st.write('')
                keep_empty_values_Duration = st.checkbox("Keep rows with empty values of Duration?")
                if keep_empty_values_Duration:  # Duration
                    min_Duration, max_Duration = return_result(cleaned_df_drop, 'Length', Length_range, 'Duration',
                                                               check=True)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_2_2_2_1_1','download_2_2_2_1_1','2download_2_2_2_1_1', cleaned_df_drop)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_drop, 'Duration', Duration_range))

                else:  # Duration
                    min_Duration, max_Duration, cleaned_df_Duration, df_with_empty_values_Duration = return_result(
                        cleaned_df_drop, 'Length',
                        Length_range,
                        'Duration',
                        check=False)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_2_2_2_1_2','download_2_2_2_1_2','2download_2_2_2_1_2', cleaned_df_Duration,
                                df_with_empty_values_Duration)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    st.write(filter_rows_by_range(cleaned_df_Duration, 'Duration', Duration_range))




            else:  # Length
                min_Length, max_Length, cleaned_df_Length, df_with_empty_values_Length = return_result(cleaned_df_drop,
                                                                                                       'Drop',
                                                                                                       drop_range,
                                                                                                       'Length',
                                                                                                       check=False)
                Length_range = st.slider('Select the range of Length', min_value=min_Length, max_value=max_Length,
                                         value=(min_Length, max_Length))

                data_to_see('df_2_2_2_2','download_2_2_2_2','2download_2_2_2_2', cleaned_df_Length,
                            df_with_empty_values_Length)

                st.write('')
                st.write('')
                st.write('')
                keep_empty_values_Duration = st.checkbox("Keep rows with empty values of Duration?")
                if keep_empty_values_Duration:  # Duration
                    min_Duration, max_Duration = return_result(cleaned_df_Length, 'Length', Length_range, 'Duration',
                                                               check=True)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_2_2_2_2_1','download_2_2_2_2_1','2download_2_2_2_2_1', cleaned_df_Length)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_Length, 'Duration', Duration_range))

                else:  # Duration
                    min_Duration, max_Duration, cleaned_df_Duration, df_with_empty_values_Duration = return_result(
                        cleaned_df_Length, 'Length',
                        Length_range,
                        'Duration',
                        check=False)
                    Duration_range = st.slider('Select the range of Duration', min_value=min_Duration,
                                               max_value=max_Duration,
                                               value=(min_Duration, max_Duration))

                    data_to_see('df_2_2_2_2_2','download_2_2_2_2_2','2download_2_2_2_2_2', cleaned_df_Duration,
                                df_with_empty_values_Duration)

                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('')
                    st.write('Final Result:')
                    df_final = (filter_rows_by_range(cleaned_df_Duration, 'Duration', Duration_range))


st.write(df_final)
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')

csv = convert_df(df_final)

# streamlit button widget
st.download_button(
   "Press to Download",
   csv,
   "file.csv",
   "text/csv",
   key='download-csv'
)

