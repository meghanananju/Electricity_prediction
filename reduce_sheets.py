""" 

import pandas as pd

# Replace 'C:\new\new\ELEC_030424\Yes_bank_prediction\combined_data_2013_to_2023.xlsx' with your file path
file_path = r'C:\new\new\Prediction_model\combined_data_2013_to_2023.xlsx'

# Read all sheets from the Excel file
excel_data = pd.read_excel(file_path, sheet_name=None)

# Filter sheets with last two numbers between 18 and 24
selected_sheets = {sheet_name: sheet_data for sheet_name, sheet_data in excel_data.items() if int(sheet_name[-2:]) in range(20, 25)}

# Write filtered sheets to a new Excel file
output_file_path = 'sheets_20_to_24.xlsx'
with pd.ExcelWriter(output_file_path) as writer:
    for sheet_name, sheet_data in selected_sheets.items():
        sheet_data.to_excel(writer, sheet_name=sheet_name, index=False)

print("Sheets with names ending between 18 and 24 have been moved to 'sheets_18_to_24.xlsx'")
 """

import pandas as pd

# Replace 'C:\new\new\ELEC_030424\Yes_bank_prediction\combined_data_2013_to_2023.xlsx' with your file path
file_path = r'C:\new\new\Prediction_model\sheets_20_to_24.xlsx'

# Read all sheets from the Excel file
excel_data = pd.read_excel(file_path, sheet_name=None)

# Display each sheet
for sheet_name, sheet_data in excel_data.items():
    print("Sheet Name:", sheet_name)

num_sheets = len(excel_data)
print("Number of sheets:", num_sheets)