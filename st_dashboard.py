import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import folium
import requests
from streamlit_folium import folium_static
import datetime as dt
import geopandas as gpd
import plotly.express as px
from deep_translator import GoogleTranslator


st.set_page_config(layout="wide")

st.write(
    f"""
    <style>
    body {{
        color: white;
        background-color: #1a1a1a;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_data():
    df = pd.read_csv(
        'https://raw.githubusercontent.com/uniofexeterstudent6/csv-file/main/data_file.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df


df = load_data()

df_static = df.copy()

st.title('GLOBAL PERCEPTION ON ELECTRIC CARS')
st.text(
    'To display additional information on a country, press the "I want to examine a country further" button !'
)
st.write('\n')
st.markdown('Click on a Country to Display its Name Continuously')


if 'keep_open_counter' not in st.session_state:
    st.session_state['keep_open_counter'] = 0

if 'selbox_available' not in st.session_state:
    st.session_state['selbox_available'] = 0

if 'counter_state' not in st.session_state:
    st.session_state['counter_state'] = 0

if 'new_country' not in st.session_state:
    st.session_state['new_country'] = 'Belgium'

if 'new_country_state' not in st.session_state:
    st.session_state['new_country_state'] = 'Belgium'

if 'map_available' not in st.session_state:
    st.session_state['map_available'] = 0

counter = st.session_state['counter_state']


def year_callback():
    global new_country

    if st.session_state['keep_open_counter'] >= 0:
        st.session_state['keep_open_counter'] += 1
    else:
        st.session_state['new_country'] = new_country
        st.session_state['new_country_state'] = selected_country
        st.session_state['country_selectbox'] = st.session_state['new_country_state']


def minnt_callback():
    global new_country
    if st.session_state['keep_open_counter'] >= 0:
        st.session_state['keep_open_counter'] += 1
    else:
        st.session_state['new_country'] = new_country
        st.session_state['new_country_state'] = selected_country
        st.session_state['country_selectbox'] = st.session_state['new_country_state']


def table_callback():
    if counter > 0:
        st.session_state['keep_open_counter'] = 1


def button_callback():
    st.session_state['keep_closing_button'] = 1
    st.session_state['selbox_available'] = 1
    st.session_state['map_available'] = 1
    st.session_state['keep_open_counter'] = -1


def button_dissapear():
    st.session_state['keep_closing_button'] = 0
    st.session_state['selbox_available'] = 0
    st.session_state['map_available'] = 0
    st.session_state['keep_open_counter'] = 0


def selectbox_callback():
    global counter
    st.session_state['selbox_available'] = 1
    st.session_state['keep_closing_button'] = 1
    counter += 1
    st.session_state['counter_state'] = counter
    selected_country = st.session_state['country_selectbox']


with st.sidebar:

    st.title('Adjustments')

    year = st.slider('Slide to select starting year', min_value=2015,
                     max_value=2023, key='year_key', on_change=year_callback)

    year_value = st.session_state['year_key']

    for i in range(2):
        print(st.write('\n'))

    min_n_tweets = st.number_input(label=f'Select a minimum number of tweets per country (max 10000)',
                                   min_value=2, max_value=10000, key='minnt_key', on_change=minnt_callback)

    minnt_value = st.session_state['minnt_key']


def filter_df(df):

    number_of_tweets_per_country = df.groupby(['country'], as_index=False)['id'].count().rename(
        columns={'country': 'name', 'id': 'number_of_tweets'})

    df['year'] = df['date'].dt.year

    filter1 = number_of_tweets_per_country[number_of_tweets_per_country['number_of_tweets']
                                           >= minnt_value]['name'].values.tolist()

    filter2 = df['year'] >= year_value

    df = df[(df['country'].isin(filter1)) & filter2]

    country_polarity_avgs = (df
                             .groupby(['country'], as_index=False)['polarity_score']
                             .mean()
                             .rename(columns={'country': 'name', 'polarity_score': 'avg_polarity_score'}))

    worldmap = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    worldmap = pd.merge(
        left=worldmap, right=country_polarity_avgs, how='left', on='name')
    worldmap['avg_polarity_score'] = pd.to_numeric(
        worldmap['avg_polarity_score'], errors='coerce')
    country_geometries = worldmap[['name', 'geometry']]

    country_polarity_avgs = pd.merge(
        left=country_polarity_avgs, right=country_geometries, on='name', how='left')

    cdf = country_polarity_avgs[['name', 'avg_polarity_score']]

    geometry = country_polarity_avgs[['name', 'geometry']]

    return df, country_polarity_avgs, cdf, geometry


df, country_polarity_avgs, cdf, geometry = filter_df(df)

col1, col2 = st.columns([3, 1.25], gap='medium')

with col1:
    # Creating the World Map

    def world_map():
        geojson_url = 'https://raw.githubusercontent.com/python-visualization/folium/main/examples/data/world-countries.json'
        response = requests.get(geojson_url)
        geojson = response.json()

        # Create a blank folium world map
        m1 = folium.Map(min_zoom=1)

        bins = 50

        threshold_scale = np.linspace(cdf['avg_polarity_score'].min(),
                                      cdf['avg_polarity_score'].max(), bins+1, dtype=float)
        threshold_scale = threshold_scale.tolist()
        threshold_scale[-1] = threshold_scale[-1] + 0.01

        folium.Choropleth(
            geo_data=geojson,
            name="Choropleth Map Coloring",
            data=cdf,
            columns=["name", "avg_polarity_score"],
            key_on="feature.properties.name",
            fill_color="YlOrBr",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name="Average Polarity Score",
            bins=bins,
            threshold_scale=threshold_scale,

            highlight=True
        ).add_to(m1)  # Add choropleth layer to the map

        geojson1 = folium.GeoJson(
            geojson,
            name="Countriy Name Displayer",
            style_function=lambda x: {
                'fillColor': '#00000000',
                'color': '#00000000'
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["name"], labels=True, sticky=False),
        ).add_to(m1)  # Add county name displays

        geojson1.add_child(folium.features.GeoJsonPopup(
            fields=["name"], aliases=["Country:"]))
        geojson1.add_child(folium.features.GeoJsonTooltip(
            fields=["name"], aliases=["Country:"]))

        folium.LayerControl().add_to(m1)  # Add layer control
        return m1

    m1 = world_map()

    # Display the map with added layers
    folium_static(m1, width=700, height=475)


with col2:

    ### TABLE ###

    hide_table_row_index = """
                        <style>
                        thead tr th:first-child {display:none}
                        tbody th {display:none}
                        </style>
                        """

    st.markdown(hide_table_row_index, unsafe_allow_html=True)

    selected_radio = st.radio('Set the number of data displayed', [
        'Top Five', 'Bottom Five'], key='n_data_on_table')

    display_df = df.groupby(['country'], as_index=False)['id'].count().rename(
        columns={'country': 'name', 'id': 'number_of_tweets'})
    display_df = pd.merge(
        left=display_df, right=cdf, on='name', how='left').rename(columns={'name': 'Country', 'number_of_tweets': 'Total Tweets',
                                                                           'avg_polarity_score': 'Avg Polarity'}).sort_values(by='Avg Polarity', ascending=False)
    display_df = display_df[display_df['Total Tweets'] >= minnt_value]

    if selected_radio == None:
        None
    elif selected_radio == 'Top Five':
        st.table(display_df.head().style.set_properties(**{'text-align': 'left',
                                                           'font-size': '20px',
                                                           'background-color': '#993404',  # from the "YlOrBr" palette
                                                           'color': 'white',
                                                           'border-color': 'black',
                                                           'border-width': '1px',
                                                           'border-style': 'solid'}))
    else:
        st.table(display_df.tail().sort_values(
            by='Avg Polarity', ascending=True).style.set_properties(**{'text-align': 'left',
                                                                       'font-size': '20px',
                                                                       'background-color': '#ffffd4',  # from the "YlOrBr" palette
                                                                       'color': 'black',
                                                                       'border-color': 'black',
                                                                       'border-width': '1px',
                                                                       'border-style': 'solid'}))

        # the colors were determined from: https://loading.io/color/feature/YlOrBr-6/

        for i in range(2):
            st.write('\n')
    country_selection_button_and_counter = st.button(
        'I want to examine a country further', key='selection_button', on_click=button_callback)
    button_true = st.session_state['selection_button']

col4, col5, col6, col7 = st.columns([1, 1, 1, 0.5])

if st.session_state['selbox_available'] == 1:
    try:

        with col2:
            for i in range(2):
                st.write('\n')
            if st.session_state['keep_closing_button'] == 1:
                closing_button = st.button(
                    'Close the display below', key='closing_button', on_click=button_dissapear)
        with col1:

            available_countries = country_polarity_avgs['name'].unique()

            selected_country = st.selectbox(label='Please Select the Country You Want to Examine Further', options=available_countries,
                                            key='country_selectbox', format_func=lambda x: 'Select an option' if x == '' else x, on_change=selectbox_callback)

            new_country = st.session_state['country_selectbox']
            st.session_state['new_country_state'] = selected_country

        col8, col9 = st.columns([1.5, 2], gap='large')

        with col8:

            rankings = cdf.sort_values(
                by='avg_polarity_score', ascending=False)
            rankings['ranking'] = range(1, len(rankings)+1)
            selcol = int(rankings[rankings['name'] ==
                                  st.session_state["country_selectbox"]]['ranking'])

            st.title(st.session_state["country_selectbox"])
            rankings = cdf.sort_values(
                by='avg_polarity_score', ascending=False)
            rankings['ranking'] = range(1, len(rankings)+1)
            selcol = int(rankings[rankings['name'] ==
                                  st.session_state["country_selectbox"]]['ranking'])
            st.markdown(
                f'Ranked {selcol} out of {len(rankings)} in Average Polarity Score')

            ### Radar Chart ###

            dfs = df.copy()

            dfs.loc[(dfs['polarity_score'] < 0, 'perception')] = 'negative'
            dfs.loc[(dfs['polarity_score'] > 0, 'perception')] = 'positive'
            dfs['perception'] = dfs['perception'].fillna(
                'neutral')

            percents = pd.DataFrame(dfs.groupby(
                ['country'])['perception'].value_counts(normalize=True).unstack().fillna(0)*100)

            categories = ['     Positive',
                          'Negative', 'Neutral', '     Positive']

            count_selected = [percents[percents.index == st.session_state["country_selectbox"]]['positive'].iloc[0],
                              percents[percents.index
                                       == st.session_state["country_selectbox"]]['negative'].iloc[0],
                              percents[percents.index == st.session_state["country_selectbox"]]['neutral'].iloc[0]]
            count_selected = np.concatenate(
                (count_selected, [count_selected[0]]))

            count_not_selected = [
                percents[percents.index !=
                         st.session_state["country_selectbox"]]['positive'].mean(),
                percents[percents.index !=
                         st.session_state["country_selectbox"]]['negative'].mean(),
                percents[percents.index != st.session_state["country_selectbox"]]['neutral'].mean()]
            count_not_selected = np.concatenate(
                (count_not_selected, [count_not_selected[0]]))

            label_placement = np.linspace(
                start=0, stop=2*np.pi, num=len(count_not_selected))

            ax3 = plt.figure(figsize=(5, 6))
            plt.subplot(polar=True)
            plt.plot(label_placement, count_selected)
            plt.plot(label_placement, count_not_selected)
            lines, labels = plt.thetagrids(
                np.degrees(label_placement), labels=categories)

            plt.title(
                f'Percentages of Tweet Emotions', y=1.1, x=0.575, fontsize=20)
            plt.legend(labels=[f'{st.session_state["country_selectbox"]}',
                       'Others'], loc=(0.655, 1), borderaxespad=0.5,
                       fontsize=10, ncol=len(dfs.columns), frameon=False)
            fig3 = ax3.get_figure()
            st.pyplot(fig3)

        with col9:

            #### Country Map ####

            if st.session_state['map_available'] == 1 and st.session_state['selbox_available'] == 1:

                selected_country_df = df[df['country'] ==
                                         st.session_state["country_selectbox"]]

                selected_country_df.loc[(selected_country_df['polarity_score']
                                        < 0, 'User Perception')] = 'negative'
                selected_country_df.loc[(selected_country_df['polarity_score']
                                        > 0, 'User Perception')] = 'positive'
                selected_country_df['User Perception'] = selected_country_df['User Perception'].fillna(
                    'neutral')

                fig = px.scatter_mapbox(selected_country_df,
                                        lon=selected_country_df['longitude'],
                                        lat=selected_country_df['latitude'],
                                        color=selected_country_df['User Perception'],
                                        size=abs(
                                            selected_country_df['polarity_score'] + 0.5).round(3),
                                        color_discrete_map={
                                            'positive': 'navy',
                                            'neutral': 'fuchsia',
                                            'negative': 'darkorange'
                                        },
                                        range_color=(selected_country_df['polarity_score'].min(
                                        ), selected_country_df['polarity_score'].max()),
                                        width=640,
                                        height=527,
                                        zoom=2.5,
                                        title='Tweet Locations and Perceptions',
                                        hover_data=['User Perception', 'number_of_likes', 'polarity_score'])

                fig.update_layout(mapbox_style='open-street-map')
                fig.update_layout(
                    margin={'r': 0, 't': 50, 'l': 0, 'b': 10})

                st.write(fig)

        #### Scatter Plot ####

        fig0, ax0 = plt.subplots(figsize=(9, 2.25), facecolor='lightblue')

        q1 = selected_country_df['number_of_likes'].quantile(0.25)
        q3 = selected_country_df['number_of_likes'].quantile(0.75)
        iqr = q3-q1

        outlier_eliminated_df = selected_country_df[selected_country_df['number_of_likes'] < q3+1.5*iqr]

        sns.scatterplot(data=outlier_eliminated_df, x='date', y='number_of_likes', size=(abs(
            selected_country_df['polarity_score']) + 0.5).round(3), hue='User Perception', palette={'positive': 'navy',
                                                                                                    'neutral': 'fuchsia',
                                                                                                    'negative': 'darkorange'}, s=5000, ax=ax0)

        ax0.set_title('Likes Throughout Years', fontsize=12, y=1.13)

        handles, labels = ax0.get_legend_handles_labels()
        legend_labels = ['positive', 'neutral', 'negative']

        values = []
        for i in range(len(labels)):
            lbl = labels[i]
            if lbl in legend_labels:
                values.append(i)

        hndls, lbls = [], []
        for k in values:
            hndls.append(handles[k])
            lbls.append(labels[k])

        ax0.legend(hndls, lbls, loc=(0.385, 1.05), borderaxespad=0.5,
                   fontsize=5, ncol=len(outlier_eliminated_df.columns), frameon=False)
        ax0.set_xlabel('')
        ax0.set_ylabel('Number of Likes', fontsize=7.5)

        ticks = np.linspace(
            0, outlier_eliminated_df['number_of_likes'].max()+3, 5)
        ax0.set_yticks(ticks=np.ceil(ticks))

        ax0.xaxis.set_tick_params(labelsize=7.5)
        ax0.yaxis.set_tick_params(labelsize=7.5)

        st.pyplot(fig0)

        st.write('\n')

        col10, col11 = st.columns([1.5, 1], gap='medium')

        with col11:
            st.markdown(
                f'Tweet with the Most Likes in {st.session_state["country_selectbox"]}:')
            dfc = df[df['country'] == st.session_state["country_selectbox"]]
            dfc = dfc.sort_values(by='number_of_likes',
                                  ascending=False).head(1)
            tweet = dfc['raw_content'].iloc[0]
            st.write(tweet)
            st.write(
                f'Number of Likes {dfc["number_of_likes"].iloc[0]}, Number of Replies {dfc["number_of_replies"].iloc[0]}, Number of Retweets {dfc["number_of_retweets"].iloc[0]}')
            st.write(f'Date: {dfc["date"].dt.date.iloc[0]}')
            if dfc['detected_language'].iloc[0] != 'en':
                translator = GoogleTranslator(source='auto', target='en')
                translation = translator.translate(tweet)
                st.markdown('English Translation:')
                st.write(translation)

        with col10:

            #### Bar and Line Plots ####

            df_static_selcol = df_static[df_static['country']
                                         == st.session_state["country_selectbox"]]

            # Data for first plot

            df_static_selcol['Year'] = df_static_selcol['date'].dt.year

            count_avgs = df_static_selcol.groupby(['Year'], as_index=False)['polarity_score'].mean(
            ).rename(columns={'polarity_score': 'Average Polarity Score'})

            count_avgs.loc[(count_avgs['Average Polarity Score']
                            > 0, 'perception')] = 'positive'
            count_avgs.loc[(count_avgs['Average Polarity Score']
                            < 0, 'perception')] = 'negative'
            count_avgs = count_avgs.fillna('neutral')

            cmap = sns.color_palette('viridis_r', as_cmap=True)

            def map_color(value):
                norm_value = (value - count_avgs['Average Polarity Score'].min()) / (
                    count_avgs['Average Polarity Score'].max() - count_avgs['Average Polarity Score'].min())
                return cmap(norm_value)

            palette = [
                map_color(value) for value in count_avgs['Average Polarity Score'].unique()]

            # Data for second plot

            count_nyears = df_static_selcol.groupby(['Year'], as_index=False)[
                'id'].count().rename(columns={'id': 'total number of tweets'})
            count_nyears['Year'] = count_nyears['Year'].astype(str)

            fig2, (ax1, ax2) = plt.subplots(
                ncols=1, nrows=2, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [2.5, 1]})

            fig2.subplots_adjust(hspace=0.3)

            plt.suptitle(
                f'Polarity Through Years for {st.session_state["country_selectbox"]}', fontsize=20, y=1)

            sns.barplot(data=count_avgs, x='Year',
                        y='Average Polarity Score', bottom=-1,   ax=ax1, palette=palette, width=0.75)

            sns.lineplot(data=count_nyears, x='Year',
                         y='total number of tweets', ax=ax2)

            ax2.xaxis.set_tick_params(labelsize=15)

            ax1.set_xlabel('')
            ax2.set_xlabel('')

            ax1.set_ylabel('')
            ax2.set_ylabel('')

            ax1.set_title('Average Polarity Score', fontsize=15,
                          loc='left', fontstyle='italic')
            ax2.set_title('Number of Tweets ', fontsize=15,
                          loc='left', fontstyle='italic')

            ax1.yaxis.set_tick_params(labelsize=10)
            ax2.yaxis.set_tick_params(labelsize=10)

            for rect in ax1.patches:
                bottom = rect.get_y()
                height = rect.get_height()
                top = bottom + height
                rect.set_height(top - bottom - (-1))

            ax1.set_ylim(-1, None)
            ax1.set_yticks(
                ticks=np.arange(-1, 1.01, 0.5))

            ax1.axhline(0, color='deeppink', linestyle='--', linewidth=1)

            sm = plt.cm.ScalarMappable(cmap=cmap)
            sm.set_array([])
            ax1.figure.colorbar(sm, ax=ax1, format="", orientation='horizontal',
                                location='top', shrink=0.5, fraction=0.1, aspect=40, anchor=(1, -0.3))

            st.pyplot(fig2)

    except ValueError:

        st.exception(
            'Error: Country not available within the parameters.\n Please press the "Close the display below" button and select a new country.')
