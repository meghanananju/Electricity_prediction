
Max Demand Analysis and Prediction
Project Overview



This project focuses on analyzing and predicting the maximum electricity demand for various states in India. The project uses historical data from 2013 to 2023 to train machine learning models, including ARIMA and Bidirectional RNN, to forecast future demand.

Directory Structure

├── arima_final.py                 # Script for ARIMA model analysis and prediction
├── Bi_rnn_final.py                # Script for Bidirectional RNN model analysis and prediction
├── combined_data_2013_to_2023.xlsx # Combined historical data for demand analysis
├── demand_trends_data.csv         # Additional demand trends data
├── pop.xlsx                       # Population data (if applicable)
├── readme.md                      # This readme file
├── reduce_sheets.py               # Script to reduce and combine Excel sheets
├── sheets_18_to_24.xlsx           # Excel sheets containing historical data from 2018 to 2024
└── sheets_20_to_24.xlsx           # Excel sheets containing historical data from 2020 to 2024
Prerequisites
Before running any scripts, make sure you have the following packages installed:

bash
pip install streamlit pandas plotly statsmodels tensorflow openpyxl


Scripts
arima_final.py
This script performs ARIMA-based analysis and prediction for electricity demand.

Functionality:

Reads historical data from the provided Excel file.
Trains the ARIMA model.
Provides options to predict demand for the next 1 month, 3 months, or 1 year.
Visualizes the predicted demand using Plotly.
Usage:
Run the script using Streamlit:

bash
 
streamlit run arima_final.py
Bi_rnn_final.py
This script performs analysis and prediction using Bidirectional RNN.

Functionality:

Reads historical data from the provided Excel file.
Trains the Bidirectional RNN model.
Provides options to predict demand for the next 1 month, 3 months, or 1 year.
Visualizes the predicted demand using Plotly.
Usage:
Run the script using Streamlit:

bash
 
streamlit run Bi_rnn_final.py
combined_data_2013_to_2023.xlsx
This Excel file contains combined historical data for electricity demand analysis from 2013 to 2023.

demand_trends_data.csv
This CSV file contains eletricity demand trends data over the years present in dataset.

pop.xlsx
This Excel file contains population data, which might be used for population demand analysis.

reduce_sheets.py
This script reduces and combines Excel sheets into a single file.

Functionality:

Reads multiple Excel sheets.
Combines them into a single sheet.
Saves the combined data into a new Excel file.
Usage:
Run the script:

bash
 
python reduce_sheets.py
sheets_18_to_24.xlsx & sheets_20_to_24.xlsx
These Excel files contain historical data for electricity demand from 2018 to 2024 and 2020 to 2024, respectively.

Instructions
Data Preparation: Ensure that the combined_data_2013_to_2023.xlsx file is up to date and contains all the necessary historical data.
Running ARIMA Model:
Open a terminal.
Run the following command:
bash
 
streamlit run arima_final.py
Running Bidirectional RNN Model:
Open a terminal.
Run the following command:
bash
 
streamlit run Bi_rnn_final.py
Combining Excel Sheets (if needed):
Ensure sheets_18_to_24.xlsx and sheets_20_to_24.xlsx are in the same directory.
Run the following command:
bash
 
python reduce_sheets.py
Notes
The prediction accuracy of the models depends on the quality and quantity of historical data provided.
Ensure that all dependencies are installed and updated to avoid compatibility issues.
Modify the script parameters as needed to fit specific requirements.
For any issues or contributions, please create an issue or a pull request on the project's GitHub repository.