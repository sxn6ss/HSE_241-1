import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.subplots as sp
import plotly.express as px
import io
import requests
from api import add_car


st.title("DATA analysis project")
st.title("Creator : Sergeev Nikita 241-1")

api_url = "http://127.0.0.1:8000"
data = pd.read_csv("car_data.csv")

st.header("Search for cars")
brand = st.text_input("Car brand:")
min_hp = st.number_input("Minimum power (h.p.):", value=0)
max_hp = st.number_input("Maximum power (h.p.):", value=500)
limit = st.number_input("How many ads to show", value=100, min_value=1)


if st.button("Get data about cars"):
    params = {"brand": brand,"min_hp": min_hp, "max_hp": max_hp, "limit": limit}
    response = requests.get(f"{api_url}/cars/", params=params)
    if response.status_code == 200:
        st.table(response.json())
    else:
        st.error(f"Error: {response.json()['detail']}")

st.header("Add new car")
new_car = {
    "car_brand": st.text_input("Car brand:", key="car_brand"),
    "car_model": st.text_input("Car model:", key="car_model"),
    "car_price": st.number_input("Car price:", value=0),
    "car_age": st.number_input("Age of the car:", value=0),
    "car_mileage": st.number_input("Mileage:", value=0),
    "car_engine_hp": st.number_input("Engine power (h.p.):", value=0),
    "car_fuel": st.text_input("Fuel type:", key="car_fuel"),
    "car_transmission": st.text_input("Transmission type:", key="car_transmission"),
    "car_country": st.text_input("Car country:", key="car_country"),
}

if st.button("Add car"):
    response = add_car(new_car)
    st.success(response["message"])
    st.write("Added car:")
    st.table(pd.DataFrame([response["car"]]))

    st.write("Updated data:")
    st.table(data.tail(10))

st.table(data.head(10))
buffer = io.StringIO()
data.info(buf=buffer)
st.text(buffer.getvalue())
st.write("All column headers:")
st.table(data.columns)
st.write("Let's check the data for gaps by calling a set of methods to summarize the missing values.")
st.table(data.isnull().sum())
st.write("Проверяем датасет на наличие дубликатов,We check the dataset for duplicates, if any, delete them. в случае наличия удаляем их.")
st.write("Repetitions found: ", data.duplicated().sum())
st.write("Since there are duplicates, you need to delete them.")
data = data.drop_duplicates().reset_index(drop=True)
st.write("We check if there are any duplicates left after deletion.")
st.write("Duplicates: ", data.duplicated().sum())
st.write("Let's add two new columns: price_per_hp — the price of a car per horsepower and age_to_mileage_ratio — the ratio of the age of the car to mileage.")
data['price_per_hp'] = data['car_price'] / data['car_engine_hp']
data['age_to_mileage_ratio'] = data['car_age'] / data['car_mileage']
st.table(data.head(10))
st.write("Now there are two additional columns in our dataset.")
st.write("Let's build graphs for two new columns.")
plt.figure(figsize=(8, 4))
plt.scatter(data["car_engine_hp"], data["price_per_hp"], label="Price per HP", color='blue')
plt.title("Cost per one h.p.")
plt.ylabel("Price per HP")
plt.legend()
plt.grid()
st.pyplot(plt)
plt.close()

plt.figure(figsize=(8, 4))
plt.scatter(data['car_mileage'], data["age_to_mileage_ratio"], label="Age to Mileage Ratio", color='green')
plt.title("The ratio of age to mileage")
plt.ylabel("Age to Mileage Ratio")
plt.legend()
plt.grid()
st.pyplot(plt)
plt.close()

st.write("We will find the most popular car for sale.")
inf_model = data[['car_brand', 'car_model']]

st.write("Dataset with information about the brand and make of the car:")
st.table(inf_model.head(10))

st.write("Now let's count the number of cars sold for each model.")
model_counts = inf_model[['car_brand', 'car_model']].value_counts().reset_index()
model_counts.columns = ['car_brand', 'car_model', 'count']
st.table(model_counts.head(10))

st.write("Now, based on these data, we will build a graph for the 10 most popular cars for sale.")
top_models = model_counts.nlargest(10, 'count')
sns.barplot(data=top_models, x='count', y='car_model', hue='car_brand', dodge=False)


plt.title('Top 10 Car Models and Brands')
plt.xlabel('Number of Occurrences')
plt.ylabel('Car Model')

st.pyplot(plt)
plt.close()

st.write("This chart shows that the Hyundai Solaris is the best-selling car.")

st.write("Let's find out which part of the car market is occupied by the manufacturing countries.")
st.write("To begin with, we will collect data that will show how many cars are sold from a particular country.")
inf_about_country = data['car_country'].value_counts()
st.table(inf_about_country.head(10))
st.write("Let's add a column to the data containing the percentage of the number of cars from specific countries. And also add a column with the count header, which will store the number of cars from each country.")
inf_about_country = data['car_country'].value_counts().reset_index()
inf_about_country.columns = ['car_country', 'count']
inf_about_country['percentage'] = (inf_about_country['count'] / inf_about_country['count'].sum()) * 100
st.table(inf_about_country.head(10))
st.write("Now let's draw a graph in the form of a map, but first you need to convert all two-letter entries to three-letter ones for proper rendering.")
corrections_countries = {
    "JP": "JPN",
    "KR": "KOR",
    "DE": "DEU",
    "RUS": "RUS",
    "USA": "USA",
    "CN": "CHN",
    "FR": "FRA",
    "CZ": "CZE",
    "UK": "GBR",
    "SE": "SWE",
    "IT": "ITA",
    "ES": "ESP",
    "UZ": "UZB",
    "IR": "IRN",
    "UKR": "UKR",
}
st.table(corrections_countries)
st.write("Now let's replace all the abbreviations with the redone ones.")
inf_about_country['car_country'] = inf_about_country['car_country'].replace(corrections_countries)
st.write("Modified dataset about the manufacturing countries::")
st.table(inf_about_country.head(10))

fig = px.choropleth(
    inf_about_country,
    locations="car_country",
    locationmode="ISO-3",
    color="percentage",
    hover_name="car_country",
    title="Car Country Distribution",
    color_continuous_scale=px.colors.sequential.Plasma
)

st.plotly_chart(fig)
st.write("This chart clearly shows that Japan is the leader in car sales in the market.")

st.write("Let's count the number of cars by brand and find out who occupies most of the car market.")
inf_about_cars_brand = data['car_brand'].value_counts().reset_index()
inf_about_cars_brand.columns = ["car_brand", "count"]
st.table(inf_about_cars_brand.head(10))
st.write("Let's add a column to our assembled dataset, where the percentage values of the total amount will be stored.")
inf_about_cars_brand['percentage'] = (inf_about_cars_brand['count'] / inf_about_cars_brand['count'].sum()) * 100
st.table(inf_about_cars_brand.head(10))
st.write("Now, based on the collected data, we will draw a graph showing which part of the auto market each car brand from our dataset occupies.")
fig = go.Figure(go.Treemap(
    labels=inf_about_cars_brand['car_brand'],
    parents=[""] * len(inf_about_cars_brand['car_brand']),
    values=inf_about_cars_brand['count'],
    textinfo="label+value",
    customdata=inf_about_cars_brand['percentage'],
    hovertemplate=(
        "<b>%{label}</b><br>"
        "Количество: %{value}<br>" 
        "Процент: %{customdata:.2f}%<extra></extra>"
    ),
    marker=dict(
        colorscale="RdYlGn",
        reversescale=True
    )
))

fig.update_layout(
    title="Information on the number of car brands occupying the car market",
    title_font_size=20
)

st.plotly_chart(fig)
st.write("This graph shows how many cars of a particular brand are placed on the market and the percentage of the market occupied by them. And Toyota is the leader in sales.")
st.write("""Next, we will try to determine how the prices of cars are formed.
1) The dependence of the price on the age of the car.
2) The dependence of the price on the mileage of the car.
3) The dependence of the price on the engine power.
4) The dependence of the price on the type of fuel..""")

st.write("Let's assemble a dataset for each of the situations.")
price_and_age = data[['car_price', 'car_age']]
price_and_mileage = data[['car_price', 'car_mileage']]
price_and_power = data[['car_price', 'car_engine_hp']]
price_and_fuel = data[['car_price', 'car_fuel']]
show_inf_data = pd.concat([price_and_age, price_and_mileage, price_and_power, price_and_fuel], axis=1)
show_inf_data = show_inf_data.loc[:, ~((show_inf_data.columns.duplicated()) & (show_inf_data.columns == "car_price"))]
st.table(show_inf_data.head(10))
st.write("Drawing graphs and a graph of general conclusions.")
fig = sp.make_subplots(rows=1, cols=4, subplot_titles=["Price | Age", "Price | Mileage", "Price | Power", "Price | Fuel"])

fig.add_trace(go.Scatter(x=price_and_age['car_age'], y=price_and_age['car_price'], mode='markers', name="Price | Age"), row=1, col=1)

fig.add_trace(go.Scatter(x=price_and_mileage['car_mileage'], y=price_and_mileage['car_price'], mode='markers', name="Price | Mileage"), row=1, col=2)

fig.add_trace(go.Scatter(x=price_and_power['car_engine_hp'], y=price_and_power['car_price'], mode='markers', name="Price | Power"), row=1, col=3)

fuel_means = price_and_fuel.groupby("car_fuel")['car_price'].mean()
fig.add_trace(go.Bar(x=fuel_means.index, y=fuel_means.values, name="Price | Fuel"), row=1, col=4)

fig.update_layout(height=500, width=1400, title={'text' : "Price Analysis", 'x': 0.5, 'xanchor': 'center'}, showlegend=False)
st.plotly_chart(fig)
st.write("These graphs show what factors can influence the pricing of cars, for example, on average, diesel-powered cars are more expensive.")
st.title("Hypothesis:")
st.write("Is it true that Toyota cars are bought with an automatic transmission and mileage up to 70,000 more often than with a manual transmission and mileage up to 100,000?")
st.write("First, let's collect all the data about Toyota cars from our dataset.")
data_of_toyota = data[(data['car_brand'] == 'Toyota')]
st.table(data_of_toyota.head(10))
st.write("Now we have to take the mileage and gearbox information from this dataset.")
data_of_toyota = data_of_toyota[['car_transmission', 'car_mileage']]
st.table(data_of_toyota.head(10))
st.write("Let's take all the values that satisfy our conditions.")
main_inf_of_toyota = data_of_toyota.loc[((data_of_toyota['car_transmission'] == 'automatic') & (data_of_toyota['car_mileage'] <= 70000))]
contr_inf_of_toyota = data_of_toyota.loc[((data_of_toyota['car_transmission'] == 'manual') & (data_of_toyota['car_mileage'] < 100000))]
st.write("Toyota cars with automatic transmission and mileage less than 70,000:")
st.table(main_inf_of_toyota.head(10))
st.write("Toyota cars with manual transmission and mileage less than 100,000:")
st.table(contr_inf_of_toyota.head(10))
st.write("Let's find out how many cars fit our conditions for each of the two cases.")
counts = [main_inf_of_toyota.shape[0], contr_inf_of_toyota.shape[0]]
conditions = ['automatic and car_mileage < 70000 |', '| manual and car_mileage < 100000']
plt.bar(conditions, counts, color = ['blue', 'green'])
plt.ylabel('Number of cars')
plt.title('Comparing the number of machines')

st.pyplot(plt)
plt.close()
st.write('Now, based on this diagram, we can conclude that cars with an automatic transmission, with a mileage of up to 70,000, are more popular than those with a manual transmission, with a mileage of up to 100,000.')
