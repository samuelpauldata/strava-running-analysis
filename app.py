import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title='Mes courses Strava', layout='wide')
st.title('Analyse de mes courses Strava')
st.markdown('Analyse de 700+ courses personnelles extraites via API Strava')

@st.cache_data
def charger_donnees():
    df = pd.read_csv('strava_courses.csv')
    df['start_date_local'] = pd.to_datetime(df['start_date_local'])
    df['annee'] = df['start_date_local'].dt.year
    df['duree_min'] = df['moving_time'] / 60
    df['allure_min_km'] = df['duree_min'] / df['distance_km']
    df['date_str'] = df['start_date_local'].dt.strftime('%Y-%m-%d')
    df['allure_str'] = df['allure_min_km'].apply(
        lambda x: f"{int(x)}:{int((x%1)*60):02d} min/km" if pd.notna(x) else 'N/A')
    return df

df = charger_donnees()

# ---- METRIQUES ----
st.header('Statistiques generales')
col1, col2, col3, col4 = st.columns(4)
col1.metric('Total courses', f"{len(df)}")
col2.metric('Distance totale', f"{df['distance_km'].sum():.0f} km")
col3.metric('Distance moyenne', f"{df['distance_km'].mean():.1f} km")
col4.metric('Allure moyenne', f"{int(df['allure_min_km'].mean())}:{int((df['allure_min_km'].mean()%1)*60):02d} min/km")

# ---- GRAPHIQUE DISTANCE PLOTLY ----
st.header('Evolution dans le temps')
fig_dist = px.scatter(
    df, x='start_date_local', y='distance_km', color='annee',
    hover_data={'start_date_local': False, 'date_str': True,
                'distance_km': ':.1f', 'allure_str': True, 'average_heartrate': ':.0f', 'annee': False},
    labels={'distance_km': 'Distance (km)', 'date_str': 'Date',
            'allure_str': 'Allure', 'average_heartrate': 'FC moyenne'},
    title='Distance par course - survolez un point pour les details'
)
fig_dist.update_traces(marker=dict(size=6, opacity=0.7))
fig_dist.update_layout(height=450, plot_bgcolor='white')
st.plotly_chart(fig_dist, use_container_width=True)

# ---- GRAPHIQUE ALLURE PLOTLY ----
df_sorted = df.sort_values('start_date_local').copy()
df_sorted['allure_mobile'] = df_sorted['allure_min_km'].rolling(window=20).mean()
fig_allure = go.Figure()
fig_allure.add_trace(go.Scatter(
    x=df_sorted['start_date_local'], y=df_sorted['allure_min_km'],
    mode='markers', name='Allure par course',
    marker=dict(size=5, color='steelblue', opacity=0.5),
    hovertemplate='<b>%{customdata[0]}</b><br>Allure: %{customdata[1]}<br>Distance: %{customdata[2]:.1f} km<extra></extra>',
    customdata=list(zip(
        df_sorted['start_date_local'].dt.strftime('%Y-%m-%d'),
        df_sorted['allure_str'],
        df_sorted['distance_km']
    ))
))
fig_allure.add_trace(go.Scatter(
    x=df_sorted['start_date_local'], y=df_sorted['allure_mobile'],
    mode='lines', name='Moyenne mobile (20 courses)',
    line=dict(color='orangered', width=2)
))
fig_allure.update_layout(
    title="Evolution de l'allure - moyenne mobile sur 20 courses",
    yaxis_autorange='reversed', height=450, plot_bgcolor='white'
)
st.plotly_chart(fig_allure, use_container_width=True)

# ---- PROGRESSION ANNUELLE ----
st.header('Progression annuelle')
stats_annee = df.groupby('annee').agg(
    km_total=('distance_km', 'sum'),
    nb_courses=('distance_km', 'count'),
    allure_moyenne=('allure_min_km', 'mean')
).reset_index()
col1, col2 = st.columns(2)
with col1:
    fig_ann1 = px.bar(stats_annee, x='annee', y='km_total',
                      color='km_total', color_continuous_scale='Reds',
                      title='Km total par annee')
    st.plotly_chart(fig_ann1, use_container_width=True)
with col2:
    fig_ann2 = px.bar(stats_annee, x='annee', y='allure_moyenne',
                      color='allure_moyenne', color_continuous_scale='Blues_r',
                      title='Allure moyenne par annee')
    fig_ann2.update_layout(yaxis_autorange='reversed')
    st.plotly_chart(fig_ann2, use_container_width=True)

# ---- CARTE GPS ----
st.header('Carte de mes courses')
st.markdown('596 courses tracees sur la carte (112 tapis roulant sans GPS)')
@st.cache_data
def charger_gps():
    return pd.read_csv('strava_gps.csv')
df_gps = charger_gps()
carte = folium.Map(location=[45.5322, -73.8953], zoom_start=10, tiles='OpenStreetMap')
df_gps_bounds = df_gps[df_gps['lat'].notna()]
lat_min, lat_max = df_gps_bounds['lat'].min(), df_gps_bounds['lat'].max()
lng_min, lng_max = df_gps_bounds['lng'].min(), df_gps_bounds['lng'].max()
carte.fit_bounds([[lat_min, lng_min], [lat_max, lng_max]])
for activity_id in df_gps['id'].unique():
    points = df_gps[df_gps['id'] == activity_id][['lat', 'lng']].values.tolist()
    if len(points) > 10:
        nom = df_gps[df_gps['id'] == activity_id]['nom'].iloc[0]
        date = str(df_gps[df_gps['id'] == activity_id]['date'].iloc[0])[:10]
        folium.PolyLine(points, color='#FC4C02', weight=2, opacity=0.7,
                       tooltip=f'{nom} - {date}').add_to(carte)
st_folium(carte, width=1200, height=500)

# ---- SIMULATEUR ML ----
st.header('Simulateur d allure - Modele ML')
@st.cache_resource
def entrainer_modele():
    df_ml = df[['distance_km','average_heartrate','total_elevation_gain','annee','allure_min_km']].dropna()
    X = df_ml[['distance_km','average_heartrate','total_elevation_gain','annee']]
    y = df_ml['allure_min_km']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modele = RandomForestRegressor(n_estimators=100, random_state=42)
    modele.fit(X_train, y_train)
    return modele
modele = entrainer_modele()
col1, col2 = st.columns(2)
with col1:
    distance = st.slider('Distance (km)', 1.0, 42.2, 10.0, 0.5)
    fc = st.slider('Frequence cardiaque (bpm)', 120, 185, 155)
with col2:
    denivele = st.slider('Denivele (m)', 0, 500, 50)
    annee = st.selectbox('Annee', [2023, 2024, 2025, 2026])
X_pred = pd.DataFrame([{'distance_km': distance, 'average_heartrate': fc,
                         'total_elevation_gain': denivele, 'annee': annee}])
allure = modele.predict(X_pred)[0]
minutes = int(allure)
secondes = int((allure - minutes) * 60)
temps_total = allure * distance
heures = int(temps_total // 60)
mins = int(temps_total % 60)
st.success(f'Allure predite : {minutes}:{secondes:02d} min/km')
if heures > 0:
    st.info(f'Temps total estime : {heures}h{mins:02d}min')
else:
    st.info(f'Temps total estime : {mins}min')
st.caption('Modele Random Forest - R2: 0.683 - MAE: 0.237 min/km')
