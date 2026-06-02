import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
    return df

df = charger_donnees()

# ---- METRIQUES ----
st.header('Statistiques generales')
col1, col2, col3, col4 = st.columns(4)
col1.metric('Total courses', f"{len(df)}")
col2.metric('Distance totale', f"{df['distance_km'].sum():.0f} km")
col3.metric('Distance moyenne', f"{df['distance_km'].mean():.1f} km")
col4.metric('Allure moyenne', f"{int(df['allure_min_km'].mean())}:{int((df['allure_min_km'].mean()%1)*60):02d} min/km")

# ---- EVOLUTION ----
st.header('Evolution dans le temps')
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(df['start_date_local'], df['distance_km'], alpha=0.5, color='orangered')
ax.set_ylabel('Distance (km)')
ax.set_title('Distance par course')
st.pyplot(fig)
plt.close()

# ---- PROGRESSION ANNUELLE ----
st.header('Progression annuelle')
stats_annee = df.groupby('annee').agg(
    km_total=('distance_km', 'sum'),
    nb_courses=('distance_km', 'count'),
    allure_moyenne=('allure_min_km', 'mean')
).reset_index()
col1, col2 = st.columns(2)
with col1:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(stats_annee['annee'], stats_annee['km_total'], color='orangered', alpha=0.8)
    ax.set_title('Km total par annee')
    st.pyplot(fig)
    plt.close()
with col2:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(stats_annee['annee'], stats_annee['allure_moyenne'], color='steelblue', alpha=0.8)
    ax.set_title('Allure moyenne par annee')
    ax.invert_yaxis()
    st.pyplot(fig)
    plt.close()

# ---- CARTE GPS ----
st.header('Carte de mes courses')
st.markdown('50 dernieres courses tracees sur la carte')

@st.cache_data
def charger_gps():
    return pd.read_csv('strava_gps.csv')

df_gps = charger_gps()
df_gps_bounds = df_gps[df_gps['lat'].notna()]
lat_min, lat_max = df_gps_bounds['lat'].min(), df_gps_bounds['lat'].max()
lng_min, lng_max = df_gps_bounds['lng'].min(), df_gps_bounds['lng'].max()
carte = folium.Map(location=[df_gps_bounds['lat'].mean(), df_gps_bounds['lng'].mean()], zoom_start=9, tiles='CartoDB dark_matter')
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
