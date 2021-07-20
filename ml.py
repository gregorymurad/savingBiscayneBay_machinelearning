# importing the required packaged in python
import pandas as pd
import os

dir_name = "All_Data/"

# specifying an empty list for content
content = []
for file in os.listdir(dir_name):
  print(file)
  # reading content into data frame
  df = pd.read_csv(dir_name + file)
  print(type(df))
  print("shape: ", df.shape)
  content.append(df)

final_content = content[0]
for df in range(1, len(content)):
  final_content = final_content.append(content[df])
print(final_content)

allDataPath = 'All_Data/biscayne_all_data.csv'
# final_content.to_csv('dir_name' + 'all_data.csv', index = False, header=True)
# remove file if exist
if os.path.exists(allDataPath):
    os.remove(allDataPath)
final_content.to_csv(allDataPath, index = False, header=True)

df_all = pd.read_csv('All_Data/biscayne_all_data.csv')

df_all = df_all[
  ["Latitude", "Longitude", "Time", "Date", "Total Water Column (m)", "Salinity (ppt)", "Temp C", "pH", "ODO mg/L"]]

df_filtered = df_all[df_all['Salinity (ppt)'] > 25]

df_filtered = df_filtered[df_filtered['Salinity (ppt)'] < 38]

df_filtered = df_filtered.reset_index()

df_filtered['Date'] = pd.to_datetime(df_filtered['Date'], format='%m/%d/%y')
# df_season = df_filtered2
df_filtered['Month'] = df_filtered['Date'].dt.month
month_dictionary = {1: 2, 2: 2, 3: 2,
                    6: 0,
                    8: 1, 9: 1}

# Add a new column named 'Price'
df_filtered['Target'] = df_filtered['Month'].map(month_dictionary)

df_filtered=df_filtered.reset_index()

print(df_filtered.describe())

filteredDataPath = 'All_Data/biscayne_all_data_filtered.csv'
# remove file if exist
if os.path.exists(filteredDataPath):
    os.remove(filteredDataPath)
df_filtered.to_csv(filteredDataPath, index = False, header=True)