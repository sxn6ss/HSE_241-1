import io
import dataframe_image as dfi
from scipy import stats as st
import asyncio
import logging
import sys
from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram import F
from aiogram.types import Message, FSInputFile, KeyboardButton, ReplyKeyboardMarkup, ReplyKeyboardRemove
import plotly.graph_objects as go
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.subplots as sp
import plotly.express as px

TOKEN = "7816338710:AAGUEH1urx6N6YgeZeAhOQibasVVL7fV7WU"

dp = Dispatcher()



df = pd.read_csv("car_data.csv")

compares = []
button1 = KeyboardButton(text="Info about DATA")
button2 = KeyboardButton(text="Show data")
button3 = KeyboardButton(text="Check hypothesis")
button4 = KeyboardButton(text="Addings to the DATA")
ex = KeyboardButton(text="Back to menu")
menu = ReplyKeyboardMarkup(resize_keyboard=True, keyboard=[[button1, button2, button3, button4]])
b1 = KeyboardButton(text="Check 1")
b2 = KeyboardButton(text="Check 2")
b3 = KeyboardButton(text="Check 3")
hypos = ReplyKeyboardMarkup(resize_keyboard=True, keyboard=[[b1, b2, b3], [ex]])
but1 = KeyboardButton(text="Graphic 1")
but2 = KeyboardButton(text="Graphic 2")
but3 = KeyboardButton(text="Graphic 3")
but4 = KeyboardButton(text="Graphic 4")
but5 = KeyboardButton(text="Graphic 5")
but6 = KeyboardButton(text="Graphic 6")
but7 = KeyboardButton(text="Graphic 7")
datas = ReplyKeyboardMarkup(resize_keyboard=True, keyboard=[[but1, but2, but3, but4, but5, but6, but7], [ex]])
but6 = KeyboardButton(text="is it true?")
grtypes = ReplyKeyboardMarkup(resize_keyboard=True, keyboard=[[but6], [ex]])
but6 = KeyboardButton(text="is it true?")
forms =ReplyKeyboardMarkup(resize_keyboard=True, keyboard=[[but6], [ex]])




@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    info = '''This project is made by Sergeev Nikita  group (241-1). 
This bot can make graphics connected to my DATA-set (Dataset of USED CARS), can show some interesting statistics and check hypotisis. 

This dataset contains information about the 'Dataset of USED CARS'. The dataset contains 12 columns and 42089 rows.

The column description of the dataset is as follows:

1) car_brand: a company that creates cars
2) car_model: the specific model of the car
3) car_price: the cost of the car
4) car_city: the city where the car is sold
5) car_fuel: the fuel that the car uses
6) car_transmission: the transmission that is used on the car
7) car_drive: drive on the car
8) car_mileage: car mileage
9) car_country: the country in which the car was assembled
10) car_engine_capacity: engine capacity in the car
11) car_engine_hp: car engine power
12) car_age: how many years have passed since the car was released
13) price_per_hp.new: the cost of a car per horsepower
14) age_to_mileage_ratio.new: the ratio of the age of the car to mileage'''

    await message.answer(info)
    dfi.export(df.head(5), "start.png")
    photo = FSInputFile("start.png")
    await message.answer_photo(photo=photo, caption="Part of DataFrame", reply_markup=menu)


@dp.message(F.text.lower() == "back to menu")
async def back_to_menu(message: Message) -> None:
    await message.answer("Menu", reply_markup=menu)



@dp.message(F.text == "Info about DATA")
async def get_stats(message: Message) -> None:

    buffer = io.StringIO()
    df.info(buf=buffer)
    s = buffer.getvalue()
    lines = [line.split() for line in s.splitlines()[3:-2]]
    info_df = pd.DataFrame(lines)
    dfi.export(info_df, "info.png")
    describe_df = df.describe()
    dfi.export(describe_df, "describe.png")
    photo1 = FSInputFile("info.png")
    await message.answer_photo(photo=photo1, caption="Info")
    photo2 = FSInputFile("describe.png")
    await message.answer_photo(photo=photo2, caption="Describe")



data = pd.read_csv("car_data.csv")
data = data.drop_duplicates().reset_index(drop=True)
data['price_per_hp'] = data['car_price'] / data['car_engine_hp']
data['age_to_mileage_ratio'] = data['car_age'] / data['car_mileage']

@dp.message(F.text.lower() == "addings to the data")
async def addings(message: Message) -> None:

    buffer = io.StringIO()
    data.info(buf=buffer)
    s = buffer.getvalue()
    lines = [line.split() for line in s.splitlines()[3:-2]]
    info_df_11 = pd.DataFrame(lines)
    dfi.export(info_df_11, "info1.png")
    describe_df_11 = data.describe()
    dfi.export(describe_df_11, "describe1.png")
    photo1 = FSInputFile("info1.png")
    await message.answer_photo(photo=photo1, caption=" New Info")
    photo2 = FSInputFile("describe1.png")
    await message.answer_photo(photo=photo2, caption="New Describe")
    info1 = ''' To the DATA I add 2 new columns( price_per_hp, age_to_mileage_ratio)
    1)price_per_hp - the cost of a car per horsepower
    2)age_to_mileage_ratio - deviation from the average by CupEquivalentSize
    '''
    await message.answer(info1)
    dfi.export(data.head(5), "start.png")
    photo = FSInputFile("start.png")
    await message.answer_photo(photo=photo, caption="Part of new DataFrame", reply_markup=menu)


def make_series(df: pd.DataFrame, column_name: str, grouping_name: str) -> pd.Series:
    return df.loc[df[column_name] == grouping_name, "RetailPrice"]

@dp.message(F.text.lower() == "show data")
async def show_data(message: Message) -> None:
    await message.answer("1) Price per horsepowe \n2) The ratio of the car's age to its mileage\n3) The most popular car for sale\n4) The market share of different countries in the car industry\n5) The most popular car manufacturer\n6) The relationship between price and various factors\n7)Hypothesis\n ", reply_markup=datas)


@dp.message(F.text.lower() == "check hypothesis")
async def check_hypo(message: Message) -> None:
    await message.answer("Our hypothesis suggests that cars with automatic transmissions and mileage up to 70,000 kilometers are more popular than cars with manual transmissions and mileage up to 100,000 kilometers. This analysis aims to statistically test whether this assumption holds true based on the available data", reply_markup=forms)

@dp.message(F.text.lower().split()[0] == "form")
async def add_to_check(message: Message) -> None:
    compares.append(message.text.split()[1])
    if len(compares) % 2 == 1:
        await message.answer("Input second form:", reply_markup=forms)
    else:
        ans = check_hypothesis(make_series(df, "Form", compares[-2]), make_series(df, "Form", compares[-1]))
        await message.answer(ans, reply_markup=menu)


@dp.message(F.text.lower() == "graphic 1")
async def print_gr1(message: Message) -> None:
    plt.figure(figsize=(8, 4))
    plt.scatter(data["car_engine_hp"], data["price_per_hp"], label="Price per HP", color='blue')
    plt.title("Price for hp")
    plt.xlabel("HP")
    plt.ylabel("Price")
    plt.legend()
    plt.grid()
    plt.savefig("horsepow.png")
    plt.clf()
    photo = FSInputFile("horsepow.png")
    caption_text = "The graph shows the relationship between price and horsepower."
    await message.answer_photo(photo=photo, caption=caption_text, reply_markup=datas)

@dp.message(F.text.lower() == "graphic 2")
async def print_gr2(message: Message) -> None:
    plt.figure(figsize=(8, 4))
    plt.scatter(data['car_mileage'], data["age_to_mileage_ratio"], label="Age to Mileage Ratio", color='green')
    plt.title("Age to Mileage Ratio")
    plt.legend()
    plt.grid()
    plt.savefig("relationage.png")
    plt.clf()
    photo = FSInputFile("relationage.png")
    caption_text = "The graph shows the relationship between age and Mileage Ratio."
    await message.answer_photo(photo=photo, caption=caption_text, reply_markup=datas)

@dp.message(F.text.lower() == "graphic 3")
async def print_gr3(message: Message) -> None:
    inf_model = data[['car_brand', 'car_model']]
    model_counts = inf_model[['car_brand', 'car_model']].value_counts().reset_index()
    model_counts.columns = ['car_brand', 'car_model', 'count']
    top_models = model_counts.nlargest(10, 'count')
    sns.barplot(data=top_models, x='count', y='car_model', hue='car_brand', dodge=False)
    
    plt.title('Top 10 Car Models and Brands')
    plt.xlabel('Number of Occurrences')
    plt.ylabel('Car Model')
    plt.savefig("top10.png")
    plt.clf()
    photo = FSInputFile("top10.png")
    caption_text = "The graph displays the top 10 most popular cars for sale."
    await message.answer_photo(photo=photo, caption=caption_text, reply_markup=datas)


@dp.message(F.text.lower() == "graphic 4")
async def print_gr4(message: Message) -> None:
    inf_about_country = data['car_country'].value_counts().reset_index()
    inf_about_country.columns = ['car_country', 'count']
    inf_about_country['percentage'] = (inf_about_country['count'] / inf_about_country['count'].sum()) * 100
    a = {
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
    inf_about_country['car_country'] = inf_about_country['car_country'].replace(a)
    sns = px.choropleth(
        inf_about_country,
        locations="car_country",
        locationmode="ISO-3",
        color="percentage",
        hover_name="car_country",
        title="Car Country Distribution",
        color_continuous_scale=px.colors.sequential.Plasma
    )
    sns.savefig("m1ap.png")
    sns.clf()
    photo = FSInputFile("m1ap.png")
    caption_text = "Car country distribution graphic. This map clearly shows that Japan leads in car sales in the market. "
    await message.answer_photo(photo=photo, caption=caption_text, reply_markup=datas)

@dp.message(F.text.lower() == "graphic 5")
async def print_gr5(message: Message) -> None:
    inf_about_cars_brand = data['car_brand'].value_counts().reset_index()
    inf_about_cars_brand.columns = ["car_brand", "count"]
    inf_about_cars_brand['percentage'] = (inf_about_cars_brand['count'] / inf_about_cars_brand['count'].sum()) * 100
    fig1 = go.Figure(go.Treemap(
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

    fig1.update_layout(
        title="Information on the number of car brands occupying the auto market.",
        title_font_size=20
    )
    fig1.savefig("matrix.png")
    fig1.clf()
    photo = FSInputFile("matrix.png")
    caption_text = "A graph showing the market share of each car brand in our dataset."
    await message.answer_photo(photo=photo, caption=caption_text, reply_markup=datas)


@dp.message(F.text.lower() == "graphic 6")
async def print_gr5(message: Message) -> None:
    price_and_age = data[['car_price', 'car_age']]
    price_and_mileage = data[['car_price', 'car_mileage']]
    price_and_power = data[['car_price', 'car_engine_hp']]
    price_and_fuel = data[['car_price', 'car_fuel']]
    show_inf_data = pd.concat([price_and_age, price_and_mileage, price_and_power, price_and_fuel], axis=1)
    fig2 = sp.make_subplots(rows=1, cols=4,
                           subplot_titles=["Price | Age", "Price | Mileage", "Price | Power", "Price | Fuel"])

    fig2.add_trace(
        go.Scatter(x=price_and_age['car_age'], y=price_and_age['car_price'], mode='markers', name="Price | Age"), row=1,
        col=1)

    fig2.add_trace(go.Scatter(x=price_and_mileage['car_mileage'], y=price_and_mileage['car_price'], mode='markers',
                             name="Price | Mileage"), row=1, col=2)

    fig2.add_trace(go.Scatter(x=price_and_power['car_engine_hp'], y=price_and_power['car_price'], mode='markers',
                             name="Price | Power"), row=1, col=3)
    
    fuel_means = price_and_fuel.groupby("car_fuel")['car_price'].mean()
    fig2.add_trace(go.Bar(x=fuel_means.index, y=fuel_means.values, name="Price | Fuel"), row=1, col=4)

    fig2.update_layout(height=500, width=1400, title={'text': "Price Analysis", 'x': 0.5, 'xanchor': 'center'},
                      showlegend=False)

    fig2.savefig("priceanalysis.png")
    fig2.clf()
    photo = FSInputFile("priceanalysis.png")
    caption_text = "A graph showing the relationship between price and four different characteristics: age, mileage, power, and fuel."
    await message.answer_photo(photo=photo, caption=caption_text, reply_markup=datas)


@dp.message(F.text.lower() == "graphic 7")
async def print_gr5(message: Message) -> None:
    data_of_toyota = data[(data['car_brand'] == 'Toyota')]
    data_of_toyota = data_of_toyota[['car_transmission', 'car_mileage']]
    main_inf_of_toyota = data_of_toyota.loc[
        ((data_of_toyota['car_transmission'] == 'automatic') & (data_of_toyota['car_mileage'] <= 70000))]
    contr_inf_of_toyota = data_of_toyota.loc[
        ((data_of_toyota['car_transmission'] == 'manual') & (data_of_toyota['car_mileage'] < 100000))]
    counts = [main_inf_of_toyota.shape[0], contr_inf_of_toyota.shape[0]]
    conditions = ['automatic and car_mileage < 70000 |', '| manual and car_mileage < 100000']
    plt.bar(conditions, counts, color=['blue', 'green'])
    plt.ylabel('Number of cars')
    plt.title('Difference in number of cars')

    plt.savefig("diff.png")
    plt.clf()
    photo = FSInputFile("diff.png")
    caption_text = "A graph illustrating my hypothesis that cars with automatic transmission and mileage up to 70,000 are more popular than those with manual transmission and mileage up to 100,000."
    await message.answer_photo(photo=photo, caption=caption_text, reply_markup=datas)


async def main() -> None:
    bot = Bot(TOKEN, parse_mode=ParseMode.HTML)

    await dp.start_polling(bot)

def check_hypothesis(series_1: pd.Series, series_2: pd.Series, alpha=0.05) -> str:
    series_1.dropna(inplace=True)
    series_2.dropna(inplace=True)
    std1 = series_1.std()
    std2 = series_2.std()
    result = st.ttest_ind(series_1, series_2, equal_var=(std1 == std2))
    if result.pvalue < alpha:
        return "We can reject the hypothesis"
    else:
        return "We cannot reject the hypothesis"


@dp.message(F.text.lower() == "is it true?")
async def check_hypothesis_handler(message: Message) -> None:
    # Фильтруем данные Toyota
    data_of_toyota1 = data[data['car_brand'] == 'Toyota']
    data_of_toyota1 = data_of_toyota1[['car_transmission', 'car_mileage']]

    # Главная и контрольная группы
    main_inf_of_toyota = data_of_toyota1.loc[
        (data_of_toyota1['car_transmission'] == 'automatic') & (data_of_toyota1['car_mileage'] <= 70000), 'car_mileage'
    ]
    contr_inf_of_toyota = data_of_toyota1.loc[
        (data_of_toyota1['car_transmission'] == 'manual') & (data_of_toyota1['car_mileage'] < 100000), 'car_mileage'
    ]

    # Проверка гипотезы
    hypothesis_result = check_hypothesis(main_inf_of_toyota, contr_inf_of_toyota)

    # Отправка результата
    await message.answer(
        f"**Hypothesis Test Result:**\n{hypothesis_result}\n\n"
        "Based on this test, we conclude whether cars with automatic transmissions and mileage "
        "under 70,000 are more popular than manual transmission cars with mileage under 100,000.",
        parse_mode="Markdown"
    )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())