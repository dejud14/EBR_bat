import streamlit as st
import pandas as pd
from io import StringIO
from ebr_bat_curtailment import (get_dataframe, get_sun_dataframe, get_temp_dataframe,
                                 add_sun_info, add_loss_column, add_night_column,
                                 add_wind_column, add_temps_columns, calculate_loss)


def upload_and_process_data():
    st.title('EBR Bat Curtailment Loss Calculator')

    # File uploaders
    data_file = st.file_uploader("Upload Energy Data", type="csv")
    sun_data_file = st.file_uploader("Upload Sun Data", type="csv")
    temp_data_file = st.file_uploader("Upload Temperature Data", type="csv")

    # Inputs for processing parameters
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Select the start date for loss calculation", value=pd.to_datetime("2022-05-01"))
    with col2:
        end_date = st.date_input("Select the end date for loss calculation", value=pd.to_datetime("2022-06-01"))

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        wind_threshold = st.number_input("Cut-in wind speed (Lower limit)", min_value=0.0, value=5.0, step=0.1)
    with col2:
        low_temp_threshold = st.number_input("Lower temperature threshold", value=0.0, step=1.0)
    with col3:
        high_temp_threshold = st.number_input("Upper temperature threshold", value=40.0, step=1.0)
    with col4:
        standby_consumption = st.number_input("Standby consumption (kWh)", value=20.0, step=0.5)

    if st.button("Process Data"):
        if data_file and sun_data_file and temp_data_file:
            try:
                # Process data with custom functions
                df_energy = get_dataframe(data_file)
                df_sun = get_sun_dataframe(sun_data_file)
                df_temp = get_temp_dataframe(temp_data_file)

                # Show initial data and add download option
                st.write("Initial Data:", df_energy)

                # Applying filters and processing
                df_energy = add_loss_column(df_energy, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
                df_energy = add_sun_info(df_energy, df_sun)
                df_energy = add_night_column(df_energy)
                df_energy = add_wind_column(df_energy, wind_threshold)
                df_energy = add_temps_columns(df_energy, df_temp, low_temp_threshold, high_temp_threshold)

                # Show applied filters data and add download option
                st.write("Applied Filters:", df_energy)

                # Final calculation
                test_lost, test_produced = calculate_loss(df_energy, standby_consumption)

                col1, col2 = st.columns(2)
                with col1:
                    st.write("Lost Energy by Turbine",
                             (lost_by_turbine := test_lost[['Energy', 'Energy Pulled']].groupby(level=1).sum()))
                with col2:
                    st.write("Produced Energy by Turbine",
                             (produced_by_turbine := test_produced['Energy'].groupby(level=1).sum()))

                col1, col2 = st.columns(2)
                with col1:
                    st.write("Total Lost Energy",
                             pd.DataFrame(lost_by_turbine.sum().T).rename(columns={0: "Total (kWh)"}))
                with col2:
                    st.write("Total Produced Energy",
                             pd.DataFrame(pd.Series(produced_by_turbine.sum())).rename(columns={0: "Total (kWh)"},
                                                                                       index={0: 'Energy'}))

                col1, col2 = st.columns(2)
                with col1:
                    st.download_button("Download Initial Data as CSV", convert_df_to_csv(df_energy), "initial_data.csv",
                                       "text/csv", key='download-initial')
                with col2:
                    st.download_button("Download Applied Filters Data as CSV", convert_df_to_csv(df_energy),
                                       "filtered_data.csv", "text/csv", key='download-filtered')
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Please upload all required files to process the data.")


def convert_df_to_csv(df):
    """Convert a DataFrame to a CSV format string."""
    output = StringIO()
    df.to_csv(output, index=True)
    return output.getvalue()


if __name__ == "__main__":
    upload_and_process_data()
