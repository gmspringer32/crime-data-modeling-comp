import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
import seaborn as sns
import jsonpickle

st.title("Chicago District 11 Crime Prediction Tool")

# Add a markdown section for the description of the project
st.markdown(
    """
This is an interactive dashboard using crime data in Chicago District 11. 
Using a linear regression and xgboost model, we are able to predict the next week of crime after a few inputs from the user.

Enter Degrees in Celcius
"""
)

# load the models
import xgboost as xgb
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression

xgb_model = XGBRegressor()
xgb_model.load_model(
    "C:/Users/gmspr/Documents/Me/Projects/crime-data-modeling-comp/crime_prediction_tool/xgb_model.model"
)

with open(
    "C:/Users/gmspr/Documents/Me/Projects/crime-data-modeling-comp/crime_prediction_tool/linear_regression_model.json",
    "r",
) as f:
    serialized_linear_model = f.read()


linear_model = jsonpickle.decode(serialized_linear_model)


class EnsembleModel:
    def __init__(self, linear_model, xgb_model):
        self.linear_model = linear_model
        self.xgb_model = xgb_model

    def predict(self, X):
        return (self.linear_model.predict(X) + self.xgb_model.predict(X))[0]


# Get input from user about todays date
today_date = st.date_input("Enter the date", pd.to_datetime("2024-02-17"))
today_date = pd.to_datetime(today_date)
today_month = today_date.month
# calculate the day of the week
today_day_of_week = today_date.dayofweek
# Get input from user about yesters total crimes
yesterday_total = st.number_input(
    "Enter the total number of crimes yesterday", min_value=0, max_value=1000, value=13
)
# Get input from user about the forecasted weather for each day in the next week
today_weather = st.selectbox(
    "Enter the weather for today",
    [
        "Clear",
        "Freezing Drizzle/Freezing Rain",
        "Ice",
        "Overcast",
        "Partially cloudy",
        "Rain",
        "Snow",
    ],
)
tomorrow_weather = st.selectbox(
    "Enter the weather for tomorrow",
    [
        "Clear",
        "Freezing Drizzle/Freezing Rain",
        "Ice",
        "Overcast",
        "Partially cloudy",
        "Rain",
        "Snow",
    ],
)
day_3_weather = st.selectbox(
    "Enter the weather for the day after tomorrow",
    [
        "Clear",
        "Freezing Drizzle/Freezing Rain",
        "Ice",
        "Overcast",
        "Partially cloudy",
        "Rain",
        "Snow",
    ],
)
day_4_weather = st.selectbox(
    "Enter the weather for 4 days from now",
    [
        "Clear",
        "Freezing Drizzle/Freezing Rain",
        "Ice",
        "Overcast",
        "Partially cloudy",
        "Rain",
        "Snow",
    ],
)
day_5_weather = st.selectbox(
    "Enter the weather for 5 days from now",
    [
        "Clear",
        "Freezing Drizzle/Freezing Rain",
        "Ice",
        "Overcast",
        "Partially cloudy",
        "Rain",
        "Snow",
    ],
)
day_6_weather = st.selectbox(
    "Enter the weather for 6 days from now",
    [
        "Clear",
        "Freezing Drizzle/Freezing Rain",
        "Ice",
        "Overcast",
        "Partially cloudy",
        "Rain",
        "Snow",
    ],
)
day_7_weather = st.selectbox(
    "Enter the weather for 7 days from now",
    [
        "Clear",
        "Freezing Drizzle/Freezing Rain",
        "Ice",
        "Overcast",
        "Partially cloudy",
        "Rain",
        "Snow",
    ],
)
# Get user input for the max and min temperature for the next week in celcius
today_max_temp = st.number_input(
    "Enter the max temperature for today", min_value=-30, max_value=50, value=0
)
today_min_temp = st.number_input(
    "Enter the min temperature for today", min_value=-30, max_value=50, value=0
)
tomorrow_max_temp = st.number_input(
    "Enter the max temperature for tomorrow", min_value=-30, max_value=50, value=0
)
tomorrow_min_temp = st.number_input(
    "Enter the min temperature for tomorrow", min_value=-30, max_value=50, value=0
)
day_3_max_temp = st.number_input(
    "Enter the max temperature for the day after tomorrow",
    min_value=-30,
    max_value=50,
    value=0,
)
day_3_min_temp = st.number_input(
    "Enter the min temperature for the day after tomorrow",
    min_value=-30,
    max_value=50,
    value=0,
)
day_4_max_temp = st.number_input(
    "Enter the max temperature for 4 days from now",
    min_value=-30,
    max_value=50,
    value=0,
)
day_4_min_temp = st.number_input(
    "Enter the min temperature for 4 days from now",
    min_value=-30,
    max_value=50,
    value=0,
)
day_5_max_temp = st.number_input(
    "Enter the max temperature for 5 days from now",
    min_value=-30,
    max_value=50,
    value=0,
)
day_5_min_temp = st.number_input(
    "Enter the min temperature for 5 days from now",
    min_value=-30,
    max_value=50,
    value=0,
)
day_6_max_temp = st.number_input(
    "Enter the max temperature for 6 days from now",
    min_value=-30,
    max_value=50,
    value=0,
)
day_6_min_temp = st.number_input(
    "Enter the min temperature for 6 days from now",
    min_value=-30,
    max_value=50,
    value=0,
)
day_7_max_temp = st.number_input(
    "Enter the max temperature for 7 days from now",
    min_value=-30,
    max_value=50,
    value=0,
)
day_7_min_temp = st.number_input(
    "Enter the min temperature for 7 days from now",
    min_value=-30,
    max_value=50,
    value=0,
)
# Get user input for if today is New Years Eve
# new_years_eve = st.selectbox("Is today New Years Eve?", ["Yes", "No"])
# Make a dataframe with the input
input_df = pd.DataFrame(
    {
        "tempmax": [
            today_max_temp,
            tomorrow_max_temp,
            day_3_max_temp,
            day_4_max_temp,
            day_5_max_temp,
            day_6_max_temp,
            day_7_max_temp,
        ],
        "tempmin": [
            today_min_temp,
            tomorrow_min_temp,
            day_3_min_temp,
            day_4_min_temp,
            day_5_min_temp,
            day_6_min_temp,
            day_7_min_temp,
        ],
        # Make a colum for each of the weather types, and set the value to 1 if it is that weather, 0 otherwise
        "Clear": [
            1 if today_weather == "Clear" else 0,
            1 if tomorrow_weather == "Clear" else 0,
            1 if day_3_weather == "Clear" else 0,
            1 if day_4_weather == "Clear" else 0,
            1 if day_5_weather == "Clear" else 0,
            1 if day_6_weather == "Clear" else 0,
            1 if day_7_weather == "Clear" else 0,
        ],
        "Freezing Drizzle/Freezing Rain": [
            1 if today_weather == "Freezing Drizzle/Freezing Rain" else 0,
            1 if tomorrow_weather == "Freezing Drizzle/Freezing Rain" else 0,
            1 if day_3_weather == "Freezing Drizzle/Freezing Rain" else 0,
            1 if day_4_weather == "Freezing Drizzle/Freezing Rain" else 0,
            1 if day_5_weather == "Freezing Drizzle/Freezing Rain" else 0,
            1 if day_6_weather == "Freezing Drizzle/Freezing Rain" else 0,
            1 if day_7_weather == "Freezing Drizzle/Freezing Rain" else 0,
        ],
        "Ice": [
            1 if today_weather == "Ice" else 0,
            1 if tomorrow_weather == "Ice" else 0,
            1 if day_3_weather == "Ice" else 0,
            1 if day_4_weather == "Ice" else 0,
            1 if day_5_weather == "Ice" else 0,
            1 if day_6_weather == "Ice" else 0,
            1 if day_7_weather == "Ice" else 0,
        ],
        "Overcast": [
            1 if today_weather == "Overcast" else 0,
            1 if tomorrow_weather == "Overcast" else 0,
            1 if day_3_weather == "Overcast" else 0,
            1 if day_4_weather == "Overcast" else 0,
            1 if day_5_weather == "Overcast" else 0,
            1 if day_6_weather == "Overcast" else 0,
            1 if day_7_weather == "Overcast" else 0,
        ],
        "Partially cloudy": [
            1 if today_weather == "Partially cloudy" else 0,
            1 if tomorrow_weather == "Partially cloudy" else 0,
            1 if day_3_weather == "Partially cloudy" else 0,
            1 if day_4_weather == "Partially cloudy" else 0,
            1 if day_5_weather == "Partially cloudy" else 0,
            1 if day_6_weather == "Partially cloudy" else 0,
            1 if day_7_weather == "Partially cloudy" else 0,
        ],
        "Rain": [
            1 if today_weather == "Rain" else 0,
            1 if tomorrow_weather == "Rain" else 0,
            1 if day_3_weather == "Rain" else 0,
            1 if day_4_weather == "Rain" else 0,
            1 if day_5_weather == "Rain" else 0,
            1 if day_6_weather == "Rain" else 0,
            1 if day_7_weather == "Rain" else 0,
        ],
        "Snow": [
            1 if today_weather == "Snow" else 0,
            1 if tomorrow_weather == "Snow" else 0,
            1 if day_3_weather == "Snow" else 0,
            1 if day_4_weather == "Snow" else 0,
            1 if day_5_weather == "Snow" else 0,
            1 if day_6_weather == "Snow" else 0,
            1 if day_7_weather == "Snow" else 0,
        ],
        "Day of Week Sin": [
            np.sin(today_day_of_week),
            np.sin(today_day_of_week + 1),
            np.sin(today_day_of_week + 2),
            np.sin(today_day_of_week + 3),
            np.sin(today_day_of_week + 4),
            np.sin(today_day_of_week + 5),
            np.sin(today_day_of_week + 6),
        ],
        "Day of Week Cos": [
            np.cos(today_day_of_week),
            np.cos(today_day_of_week + 1),
            np.cos(today_day_of_week + 2),
            np.cos(today_day_of_week + 3),
            np.cos(today_day_of_week + 4),
            np.cos(today_day_of_week + 5),
            np.cos(today_day_of_week + 6),
        ],
        "Month Cos": [
            np.cos(today_month),
            np.cos(today_month),
            np.cos(today_month),
            np.cos(today_month),
            np.cos(today_month),
            np.cos(today_month),
            np.cos(today_month),
        ],
        "Month Sin": [
            np.sin(today_month),
            np.sin(today_month),
            np.sin(today_month),
            np.sin(today_month),
            np.sin(today_month),
            np.sin(today_month),
            np.sin(today_month),
        ],
        "New Year's Eve": np.zeros(7),
        "Yesterday Total": np.zeros(7),
    }
)

input_df = input_df[
    [
        "tempmax",
        "tempmin",
        "Day of Week Sin",
        "Day of Week Cos",
        "Clear",
        "Freezing Drizzle/Freezing Rain",
        "Ice",
        "Overcast",
        "Partially cloudy",
        "Rain",
        "Snow",
        "New Year's Eve",
        "Month Sin",
        "Month Cos",
        "Yesterday Total",
    ]
]


total_pred = yesterday_total
preds = []
ensemble_model = EnsembleModel(linear_model, xgb_model)
for i in range(len(input_df)):
    day_to_predict = input_df.iloc[i].to_frame().T
    day_to_predict["Yesterday Total"] = total_pred
    total_all_preds = ensemble_model.predict(day_to_predict)
    total_pred = total_all_preds[14]
    prev_day_pred = total_pred
    preds.append(total_all_preds)

preds = np.array(preds)

preds_total = preds[:, 14]
preds_arson = preds[:, 0]
preds_assault = preds[:, 1]
preds_battery = preds[:, 2]
preds_sexual_assualt = preds[:, 3] + preds[:, 4] + preds[:, 9]
preds_homicide = preds[:, 5]
preds_kidnapping = preds[:, 6]
preds_ritualism = preds[:, 7]
preds_robbery = preds[:, 8]
preds_afternoon = preds[:, 10]
preds_evening = preds[:, 11]
preds_morning = preds[:, 12]
preds_night = preds[:, 13]

total_calculated_type = (
    preds_arson
    + preds_assault
    + preds_battery
    + preds_sexual_assualt
    + preds_homicide
    + preds_kidnapping
    + preds_ritualism
    + preds_robbery
)
total_calculated_time = preds_morning + preds_afternoon + preds_evening + preds_night

# Show the prediction plot in streamlit

fig, ax = plt.subplots()


ax.plot(input_df.index, preds_total, label="Prediction")


# ax.set_xlim(pd.to_datetime("2024-02-18"), pd.to_datetime("2024-02-24"))


ax.legend()
ax.set_title("Predicted Total Crimes Per Day\nfor the next week")
ax.set_ylim(0, max(preds_total) + 3)


plt.xticks(rotation=45)

for point in zip(input_df.index, preds_total):
    plt.annotate(round(point[1], 2), xy=(point))

st.pyplot(fig)
# plt.show()

perc_homicide = (preds_homicide / total_calculated_type) * 100
perc_sexual_assault = (preds_sexual_assualt / total_calculated_type) * 100
perc_arson = (preds_arson / total_calculated_type) * 100
perc_assault = (preds_assault / total_calculated_type) * 100
perc_battery = (preds_battery / total_calculated_type) * 100
perc_kidnapping = (preds_kidnapping / total_calculated_type) * 100
perc_ritualism = (preds_ritualism / total_calculated_type) * 100
perc_robbery = (preds_robbery / total_calculated_type) * 100

fig, ax = plt.subplots()
plotdf = pd.DataFrame(
    {
        "Days From Today": input_df.index,
        "homicide": perc_homicide,
        "sexual assault": perc_sexual_assault,
        "arson": perc_arson,
        "assault": perc_assault,
        "battery": perc_battery,
        "kidnapping": perc_kidnapping,
        "ritualism": perc_ritualism,
        "robbery": perc_robbery,
    }
)
plotdf = plotdf.applymap(lambda x: 0 if x < 0 else x)
plotdf = plotdf.round(2)
plotdf.set_index("Days From Today", inplace=True)
plotdf.plot(kind="bar", stacked=True, ax=ax)
plt.title("Type of Violent Crime and\n Predicted Percent of Total Crimes")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
st.pyplot(fig)
plotdf = plotdf.applymap(lambda x: "{:.2f}%".format(x))
st.write("Predicted Percentage of Total Violent Crimes")
st.dataframe(plotdf)
# plt.show()
