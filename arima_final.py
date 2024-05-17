import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import datetime
import plotly.graph_objs as go
from itertools import product
import time
from joblib import Parallel, delayed
import copy
@st.cache_resource()
def read_excel_file(excel_file):
    return pd.read_excel(excel_file, sheet_name=None)

def get_historical_data_for_state(xls, state):
    data_with_dates = []
    for sheet_name, df in xls.items():
        if state in df.values:
            df_copy = copy.deepcopy(df)
            df_copy['Date'] = pd.to_datetime(sheet_name, format='%d-%m-%y')
            data_with_dates.append(df_copy[df_copy.apply(lambda row: state in row.values, axis=1)])

    state_data = pd.concat(data_with_dates)
    return state_data

def train_arima_model(data):
    numeric_data = pd.to_numeric(data.iloc[:, 3], errors='coerce').dropna()
    best_aic = np.inf
    best_order = None
    p_values = range(0, 2)
    d_values = range(0, 2)
    q_values = range(0, 2)

    for p, d, q in product(p_values, d_values, q_values):
        try:
            model = ARIMA(numeric_data, order=(p, d, q))
            results = model.fit()
            if results.aic < best_aic:
                best_aic = results.aic
                best_order = (p, d, q)
        except:
            continue

    arima_model = ARIMA(numeric_data, order=best_order)
    arima_model_fit = arima_model.fit()

    return arima_model_fit


@st.cache_resource()
def train_arima_models_parallel(selected_states, xls):
    models = {}
    models = Parallel(n_jobs=-1)(delayed(train_arima_model)(get_historical_data_for_state(xls, state)) for state in selected_states)
    return {selected_states[i]: models[i] for i in range(len(selected_states))}

def predict_demand_arima(models, steps):
    predicted_demand = {}
    for selected_states, model in models.items():
        forecast = model.forecast(steps=steps)
        predicted_demand[selected_states] = [((datetime.date.today() + datetime.timedelta(days=i)).strftime("%Y-%m-%d"), round(value)) for i, value in enumerate(forecast)]

    return predicted_demand

# Initial model training
excel_file = "combined_data_2013_to_2023.xlsx"
xls = read_excel_file(excel_file)

selected_states = ["Punjab", "Haryana", "Rajasthan", "Delhi", "UP", "Uttarakhand", "HP", "Chandigarh",
                   "Chhattisgarh", "Gujarat", "MP", "Maharashtra", "Goa", 
                  "Andhra Pradesh", "Telangana", "Karnataka", "Kerala", "Tamil Nadu",  "Bihar","Jharkhand",
                  "Odisha", "West Bengal", "Sikkim", "Arunachal Pradesh", "Assam", "Manipur", "Meghalaya",
                  "Mizoram", "Nagaland", "Tripura"]

arima_models = train_arima_models_parallel(selected_states, xls)

def main():
    options = st.sidebar.selectbox("Choose Analysis", ["Max Demand Analysis", "Per Capita Demand Analysis", "Demand Trends Analysis", "Season Wise Demand Analysis", "Week Demand Analysis"])

    if options == "Max Demand Analysis":
        max_demand_analysis()
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
        


def max_demand_analysis():
    st.title("Max Demand Analysis and Prediction")

    options = st.selectbox("Select Prediction Interval", ("","Next 1 Month", "Next 3 Months", "Next 1 Year"))

    if options:
        if options == "Next 1 Month":
            steps = 30
        elif options == "Next 3 Months":
            steps = 90
        elif options == "Next 1 Year":
            steps = 365
    else:
        return

    start_time = time.time()
    predicted_demand_arima = predict_demand_arima(arima_models, steps=steps)
    end_time = time.time()
    st.write(f"Time taken for prediction: {end_time - start_time} seconds")

    st.write(f"Predicted Maximum Demand using model for the {options}:")

    if predicted_demand_arima:
        graph_type = st.radio("Select Graph Type", ("Bar Graph", "Line Graph"), index=0)

        if graph_type == "Bar Graph":
            fig_bar = go.Figure()

            for state, demand in predicted_demand_arima.items():
                dates, values = zip(*demand)
                fig_bar.add_trace(go.Bar(x=dates, y=values, name=state, 
                                         hoverinfo='text', hovertext=[f"Date: {date}<br>Max Demand: {value}" for date, value in zip(dates, values)]))

            fig_bar.update_layout(title=f'Predicted Maximum Demand for {options} (Bar Graph)',
                                  xaxis_title='Date',
                                  yaxis_title='Max Demand')

            st.plotly_chart(fig_bar)

        elif graph_type == "Line Graph":
            fig_line = go.Figure()

            for state, demand in predicted_demand_arima.items():
                dates, values = zip(*demand)
                fig_line.add_trace(go.Scatter(x=dates, y=values, mode='lines+markers', name=state, 
                                              hoverinfo='text', hovertext=[f"Date: {date}<br>Max Demand: {value}" for date, value in zip(dates, values)]))

            fig_line.update_layout(title=f'Predicted Maximum Demand for {options} (Line Graph)',
                                   xaxis_title='Date',
                                   yaxis_title='Max Demand')

            st.plotly_chart(fig_line)
            
    
    
#---------you can use this if you want to see the week analysis of future dates--------#
# def week_demand_analysis(selected_date, predicted_demand_arima):
    
#     week_start_date = selected_date - datetime.timedelta(days=selected_date.weekday())
#     week_end_date = week_start_date + datetime.timedelta(days=6)

#     st.write(f"Week starting from: {week_start_date.strftime('%Y-%m-%d')} to {week_end_date.strftime('%Y-%m-%d')}")

#     week_data = {}
#     for state, demand in predicted_demand_arima.items():
#         state_demand = [d[1] for d in demand if datetime.datetime.strptime(d[0], "%Y-%m-%d").date() >= week_start_date.date() and datetime.datetime.strptime(d[0], "%Y-%m-%d").date() <= week_end_date.date()]
#         max_demand = max(state_demand) if state_demand else 0
#         week_data[state] = {'max_demand': max_demand}

#     # Create a bar chart for week demand analysis
#     fig_week_demand = go.Figure()

#     for state, data in week_data.items():
#         fig_week_demand.add_trace(go.Bar(x=[state], y=[data['max_demand']], name=state,
#                                          hoverinfo='text',
#                                          text=f"Max Demand: {data['max_demand']}"))

#     fig_week_demand.update_layout(title='Max Demand for each state within the Week',
#                                    xaxis_title='State',
#                                    yaxis_title='Max Demand')

#     st.plotly_chart(fig_week_demand)

#-----------week analysis of selected date---------#
# def week_demand_analysis(selected_date_str):
#     # Convert selected_date_str to datetime object
#     selected_date = datetime.datetime.strptime(selected_date_str, "%d-%m-%y")
    
#     week_start_date = selected_date - datetime.timedelta(days=selected_date.weekday())
#     week_end_date = week_start_date + datetime.timedelta(days=6)

#     st.write(f"Week starting from: {week_start_date.strftime('%Y-%m-%d')} to {week_end_date.strftime('%Y-%m-%d')}")

#     week_data = {}
#     for state in selected_states:
#         state_historical_data = get_historical_data_for_state(xls, state)
#         state_week_data = state_historical_data[(state_historical_data['Date'] >= week_start_date) & (state_historical_data['Date'] <= week_end_date)]
#         max_demand_day = determine_max_demand_day_for_week(state_week_data)
#         week_data[state] = {'max_demand_day': max_demand_day}

#     # Create a bar chart for week demand analysis
#     fig_week_demand = go.Figure()

#     for state, data in week_data.items():
#         fig_week_demand.add_trace(go.Bar(x=[state], y=[data['max_demand_day']], name=state,
#                                          hoverinfo='text',
#                                          text=f"Max Demand Day: {data['max_demand_day']}"))

#     fig_week_demand.update_layout(title='Max Demand Day for each state within the Week',
#                                    xaxis_title='State',
#                                    yaxis_title='Max Demand Day')

#     st.plotly_chart(fig_week_demand)

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
            max_demand_day = date.strftime("%d-%m-%y")

    return max_demand_day

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

    season_options = {
        "Winter": [12, 1, 2, 3],
        "Summer": [4, 5, 6],
        "Monsoon": [7, 8, 9]
    }

    selected_season = st.selectbox("Select Season", list(season_options.keys()))

    selected_months = season_options[selected_season]

    st.write(f"Selected Season: {selected_season}")

    season_data = {}

    # Define a function to compute max demand for each state
    def compute_max_demand_for_state(state):
        state_historical_data = get_historical_data_for_state(xls, state)
        state_season_data = state_historical_data[state_historical_data['Date'].dt.month.isin(selected_months)]
        state_season_data['Demand'] = pd.to_numeric(state_season_data.iloc[:, 3], errors='coerce')
        max_demand = state_season_data['Demand'].max()
        
        return state, max_demand
    

    # Parallel processing to compute max demand for all states
    results = Parallel(n_jobs=-1)(delayed(compute_max_demand_for_state)(state) for state in selected_states)

    # Collect results
    season_data = dict(results)

    max_state = max(season_data, key=season_data.get)
    st.write(f"In {selected_season}, {max_state} has too much consumption of {season_data[max_state]}.")



if __name__ == "__main__":
    main()