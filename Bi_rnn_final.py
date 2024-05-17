import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.graph_objs as go
import time
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, SimpleRNN, Dense

@st.cache_resource()
def read_excel_file(excel_file):
    return pd.read_excel(excel_file, sheet_name=None)


def train_SimpleRNN_model(data):
    # Convert the selected column to numeric datatype and handle non-numeric values
    numeric_data = pd.to_numeric(data.iloc[:, 3], errors='coerce').dropna()

    # Reshape the data to have two dimensions
    numeric_data = np.array(numeric_data).reshape(-1, 1)

    # Normalize data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(numeric_data)

    # Prepare data for SimpleRNN model
    X_train, y_train = [], []
    for i in range(100, len(scaled_data)):
        X_train.append(scaled_data[i-100:i, 0])
        y_train.append(scaled_data[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # Reshape data for SimpleRNN model
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    # Build SimpleRNN model
    model = Sequential()
    model.add(Bidirectional(SimpleRNN(units=10, return_sequences=True), input_shape=(X_train.shape[1], 1)))
    model.add(Bidirectional(SimpleRNN(units=10)))
    model.add(Dense(units=1))

    # Compile model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Fit the model to the training data
    model.fit(X_train, y_train, epochs=5, batch_size=32)

    return model, scaler

excel_file = "sheets_20_to_24.xlsx"
xls = read_excel_file(excel_file)

selected_states = ["Punjab", "Haryana", "Rajasthan", "Delhi", "UP", "Uttarakhand", "HP", "Chandigarh",
                   "Chhattisgarh", "Gujarat", "MP", "Maharashtra", "Goa", 
                  "Andhra Pradesh", "Telangana", "Karnataka", "Kerala", "Tamil Nadu",  "Bihar","Jharkhand",
                  "Odisha", "West Bengal", "Sikkim", "Arunachal Pradesh", "Assam", "Manipur", "Meghalaya",
                  "Mizoram", "Nagaland", "Tripura"]



def get_historical_data_for_state(xls, state):
    data_with_dates = []
    for sheet_name, df in xls.items():
        if state in df.values:
            df_copy = df.copy()
            df_copy['Date'] = pd.to_datetime(sheet_name, format='%d-%m-%y')
            data_with_dates.append(df_copy[df_copy.apply(lambda row: state in row.values, axis=1)])

    state_data = pd.concat(data_with_dates)
    
    return state_data

def get_historical_data_for_state_max_demand(xls, state):
    # Initialize an empty list to store data frames with dates
    data_with_dates = []

    # Iterate over each sheet
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name)
        if state in df.values:
            # Add a new column with the date from the sheet name
            df['Date'] = pd.to_datetime(sheet_name, format='%d-%m-%y')
            # Append the modified DataFrame to the list
            data_with_dates.append(df[df.apply(lambda row: state in row.values, axis=1)])

    # Concatenate all DataFrames with dates
    state_data = pd.concat(data_with_dates)

    return state_data






def view_historical_data():
    st.title("View Historical Data")

    selected_state = st.selectbox("Select State", selected_states)
    state_data = get_historical_data_for_state(xls, selected_state)


    # Convert Date column to datetime
    state_data['Date'] = pd.to_datetime(state_data['Date'])

    # Sort the DataFrame by 'Date' column
    state_data_sorted = state_data.sort_values(by='Date')

    # Filter out rows with NaN demand values
    state_data_sorted = state_data_sorted.dropna(subset=[state_data_sorted.columns[3]])

    # Create Plotly line chart
    fig = go.Figure()
    fig.add_trace(go.Bar(x=state_data_sorted['Date'], y=state_data_sorted.iloc[:, 3], name='Demand'))

    fig.update_layout(title=f'Historical Data for {selected_state}',
                      xaxis_title='Date',
                      yaxis_title='Demand')

    st.plotly_chart(fig)
def load_population_data():
    return pd.read_excel("pop.xlsx")

def generate_per_capita_demand_plot(pop_data):
    # Drop rows with NaN values
    pop_data_cleaned = pop_data.dropna(subset=['States', 'Population', 'Total Max Demand'])

    # Sort the population data based on population in ascending order
    pop_data_sorted = pop_data_cleaned.sort_values(by='Population')

    fig_per_capita_demand = go.Figure()
    for state, population, demand in zip(
            pop_data_sorted['States'], pop_data_sorted['Population'], pop_data_sorted['Total Max Demand']):
        fig_per_capita_demand.add_trace(go.Bar(x=[state], y=[population],                  
                                               hoverinfo='text',
                                               text=[f"State: {state}<br>Population: {population}<br>Demand: {demand}"],
                                               name=state  # Set state name as legend
                                               ))

    fig_per_capita_demand.update_layout(title='Per Capita Demand Analysis',
                                        xaxis_title='States',
                                        yaxis_title='Population',
                                        legend_title='States')  # Set legend title
    return fig_per_capita_demand


def per_capita_demand_analysis():
    st.title("Per Capita Demand Analysis")

    pop_data = load_population_data()

    
    fig_per_capita_demand = generate_per_capita_demand_plot(pop_data)
    st.plotly_chart(fig_per_capita_demand)

def demand_trends_analysis():
    st.title("Trends of Peak Demand Analysis")

    # Read the data from the Excel file
    xls = pd.read_csv("demand_trends_data.csv")

    # Create a bar graph
    fig_demand_trends = go.Figure()

    # Add bar traces for each state's maximum demand
    for state, data in xls.groupby('State'):
        fig_demand_trends.add_trace(go.Bar(x=data['Year'], y=data['Max Demand'], name=state))

    # Update layout
    fig_demand_trends.update_layout(title='Trends of Demand Analysis Over Every States',
                                     xaxis_title='Year',
                                     yaxis_title='Total Max Demand')

    # Display the graph
    st.plotly_chart(fig_demand_trends)
    
def season_wise_demand_analysis():
    st.title("Season Wise Demand Analysis")

    # Specify the selected states
    selected_states = ["Rajasthan", "Delhi", "UP", "Gujarat", "Maharashtra", "Andhra Pradesh",
                       "Karnataka", "Tamil Nadu", "West Bengal"]

    season_options = {
        "Winter": [12, 1, 2, 3],
        "Summer": [4, 5, 6],
        "Monsoon": [7, 8, 9]
    }

    selected_season = st.selectbox("Select Season", list(season_options.keys()))
    selected_months = season_options[selected_season]
    season_data = {}

    # Define a function to compute max demand for each state
    def compute_max_demand_for_state(state):
        state_historical_data = get_historical_data_for_state(xls, state)
        state_season_data = state_historical_data[state_historical_data['Date'].dt.month.isin(selected_months)]
        state_season_data['Demand'] = pd.to_numeric(state_season_data.iloc[:, 3], errors='coerce')
        max_demand = state_season_data['Demand'].max()
        return state, max_demand

    # Parallel processing to compute max demand for selected states
    results = Parallel(n_jobs=-1)(delayed(compute_max_demand_for_state)(state) for state in selected_states)

    # Collect results
    season_data = dict(results)

    max_state = max(season_data, key=season_data.get)
    st.write(f"In {selected_season}, {max_state} has the highest electricity consumption with demand of {season_data[max_state]}.")


# def max_demand_analysis():
#     st.title("Max Demand Analysis and Prediction")

#     options = st.selectbox("Select Prediction Interval", ("","Next 1 Month", "Next 3 Months", "Next 1 Year"))

#     if options:
#         if options == "Next 1 Month":
#             steps = 30
#         elif options == "Next 3 Months":
#             steps = 90
#         elif options == "Next 1 Year":
#             steps = 365
#     else:
#         return

#     start_time = time.time()
#     rnn_models = train_rnn_models_parallel(selected_states, xls, seq_length=30)
#     end_time = time.time()
#     st.write(f"Time taken for model training: {end_time - start_time} seconds")

#     st.write(f"Predicted Maximum Demand for the {options}:")

#     predicted_demand_rnn = {}
#     for state, model_info in rnn_models.items():
#         last_sequence = model_info['scaler'].transform(get_historical_data_for_state(xls, state)['demand'].values.reshape(-1, 1))[-30:]
#         forecast = predict_demand_rnn(model_info['model'], last_sequence, model_info['scaler'], steps)
#         predicted_demand_rnn[state] = forecast.tolist()

#     if predicted_demand_rnn:
#         # Generate graph
#         fig = go.Figure()
#         for state, demand in predicted_demand_rnn.items():
#             dates = [datetime.date.today() + datetime.timedelta(days=i) for i in range(len(demand))]
#             fig.add_trace(go.Scatter(x=dates, y=demand, mode='lines', name=state))
        
#         fig.update_layout(title=f'Predicted Maximum Demand for {options}',
#                           xaxis_title='Date',
#                           yaxis_title='Max Demand')
        
#         st.plotly_chart(fig)

def max_demand_analysis():
    
    st.title("Max Demand Analysis and Prediction")

    # Read Excel file
    excel_file = "sheets_20_to_24.xlsx"
    xls = pd.ExcelFile(excel_file)
    

    # List of hardcoded states
    states = ["Punjab", "Haryana", "Rajasthan", "Delhi", "UP", "Uttarakhand", "HP", "Chandigarh",
               "Chhattisgarh", "Gujarat", "MP", "Maharashtra", "Goa",
              "Andhra Pradesh", "Telangana", "Karnataka", "Kerala", "Tamil Nadu", "Bihar", "Jharkhand",
              "Odisha", "West Bengal", "Sikkim", "Arunachal Pradesh", "Assam", "Manipur", "Meghalaya",
              "Mizoram", "Nagaland", "Tripura"]

    # Checkbox to select all states
    select_all_states = st.checkbox("Select All States")

    if select_all_states:
        selected_states = states
    else:
        # Select states for prediction
        selected_states = st.multiselect("Select States", states)

    if not selected_states:
        st.warning("Please select at least one state.")
        return

    # Get historical data for selected states
    historical_data = pd.concat([get_historical_data_for_state_max_demand(xls, state) for state in selected_states])

    # Train SimpleRNN models
    SimpleRNN_models = {state: train_SimpleRNN_model(get_historical_data_for_state_max_demand(xls, state)) for state in selected_states}

    prediction_intervals = {"Next 1 Month": 30, "Next 3 Months": 90, "Next 1 Year": 365}

    for interval, steps in prediction_intervals.items():
        predicted_demand_SimpleRNN = predict_demand_rnn(SimpleRNN_models, steps=steps)


        if predicted_demand_SimpleRNN:
            fig_bar = go.Figure()

            for state, demand in predicted_demand_SimpleRNN.items():
                if state in selected_states:
                    dates, values = zip(*demand)
                    
                    fig_bar.add_trace(go.Bar(x=[state]*len(values), y=values, name=state, 
                                             hoverinfo='text', hovertext=[f"State: {state}<br>Max Demand: {value}" for value in values]))

            fig_bar.update_layout(title=f'Predicted Maximum Demand for {interval} (Bar Graph)',
                                  xaxis_title='State',
                                  yaxis_title='Predicted Maximum Demand')

            st.plotly_chart(fig_bar)

def predict_demand_rnn(models, steps):
    predicted_demand = {}
    for state, (model, scaler) in models.items():
        # Predict maximum demand for each day within the specified number of steps using SimpleRNN
        forecast = []
        for i in range(steps):
            # Predict demand for the next day
            input_data = np.array([[datetime.date.today().year, datetime.date.today().month, datetime.date.today().day + i]])
            predicted_value_scaled = model.predict(input_data)
            predicted_value = scaler.inverse_transform(predicted_value_scaled)
            forecast.append((datetime.date.today() + datetime.timedelta(days=i), round(predicted_value[0][0])))
        predicted_demand[state] = forecast

    return predicted_demand



def week_demand_analysis(selected_states, selected_date_str, xls):
    selected_date = datetime.datetime.strptime(selected_date_str, "%d-%m-%y")
    week_start_date = selected_date - datetime.timedelta(days=selected_date.weekday())
    week_end_date = week_start_date + datetime.timedelta(days=6)

    st.write(f"Week starting from: {week_start_date.strftime('%Y-%m-%d')} to {week_end_date.strftime('%Y-%m-%d')}")

    week_data = {}
    for state in selected_states:
        state_historical_data = get_historical_data_for_state(xls, state)
        state_week_data = state_historical_data[(state_historical_data['Date'] >= week_start_date) & (state_historical_data['Date'] <= week_end_date)]
        max_demand_day = determine_max_demand_day_for_week(state_week_data)
        week_data[state] = {'max_demand_day': max_demand_day}

    # Create a bar chart for week demand analysis
    fig_week_demand = go.Figure()

    for state, data in week_data.items():
        fig_week_demand.add_trace(go.Bar(x=[state], y=[data['max_demand_day']], name=state,
                                         hoverinfo='text',
                                         text=f"Max Demand Day: {data['max_demand_day']}"))
                                         

    fig_week_demand.update_layout(title='Max Demand Day for each state within the Week',
                                   xaxis_title='State',
                                   yaxis_title='Max Demand Day')

    st.plotly_chart(fig_week_demand)

def week_demand_analysis_ui(selected_states):
    st.title("Week Demand Analysis")
    selected_states = st.multiselect("Select States", selected_states)
    selected_date_str = st.text_input("Provide a date for Week Demand Analysis (e.g., dd-mm-yy)")
    
    if selected_states and selected_date_str:
        week_demand_analysis(selected_states, selected_date_str, xls)
def determine_max_demand_day_for_week(data):
    max_demand_day = None
    max_demand = 0

    for date, value in zip(data['Date'], data.iloc[:, 3]):
        try:
            value = int(value)
        except ValueError:
            continue
        
        if value > max_demand:
            max_demand = value
            max_demand_day = date
            

    return max_demand_day.strftime('%Y-%m-%d') 

def main():
    options = st.sidebar.selectbox("Choose Analysis", ["Max Demand Analysis","Week Demand Analysis", "View Historical Data","Per Capita Demand Analysis","Demand Trends Analysis","Season Wise Demand Analysis"])

    if options == "Max Demand Analysis":
        max_demand_analysis()
    elif options == "View Historical Data":
        view_historical_data()
    elif options == "Per Capita Demand Analysis":
        per_capita_demand_analysis()
    elif options == "Demand Trends Analysis":
        demand_trends_analysis()
    elif options == "Season Wise Demand Analysis":
        season_wise_demand_analysis()
    elif options == "Week Demand Analysis":
        selected_states = ["Punjab", "Haryana", "Rajasthan", "Delhi", "UP", "Uttarakhand", "HP", "Chandigarh",
                   "Chhattisgarh", "Gujarat", "MP", "Maharashtra", "Goa", 
                  "Andhra Pradesh", "Telangana", "Karnataka", "Kerala", "Tamil Nadu",  "Bihar","Jharkhand",
                  "Odisha", "West Bengal", "Sikkim", "Arunachal Pradesh", "Assam", "Manipur", "Meghalaya",
                  "Mizoram", "Nagaland", "Tripura"]
        week_demand_analysis_ui(selected_states)


if __name__ == "__main__":
    main()
