import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title='Prevision vs Realise', layout='wide')
st.title('Prevision ML vs Performances reelles')
st.markdown('Comparaison entre les predictions du modele Random Forest et mes vrais temps de course')

@st.cache_data
def charger_donnees():
    df = pd.read_csv('strava_courses.csv')
    df['start_date_local'] = pd.to_datetime(df['start_date_local'])
    df['annee'] = df['start_date_local'].dt.year
    df['duree_min'] = df['moving_time'] / 60
    df['allure_min_km'] = df['duree_min'] / df['distance_km']
    df['mois'] = df['start_date_local'].dt.month
    df['jour_semaine'] = df['start_date_local'].dt.dayofweek
    return df

@st.cache_resource
def entrainer_modele(df):
    df_ml = df[['distance_km', 'average_heartrate', 'total_elevation_gain',
                'annee', 'mois', 'jour_semaine', 'allure_min_km']].dropna()
    X = df_ml[['distance_km', 'average_heartrate', 'total_elevation_gain',
               'annee', 'mois', 'jour_semaine']]
    y = df_ml['allure_min_km']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modele = RandomForestRegressor(n_estimators=100, random_state=42)
    modele.fit(X_train, y_train)
    return modele

df = charger_donnees()
modele = entrainer_modele(df)

# Performances reelles
competitions = [
    {'distance': '5 km',          'distance_km': 5.0,    'temps_reel': '18:56', 'temps_min': 18 + 56/60},
    {'distance': '10 km',         'distance_km': 10.0,   'temps_reel': '38:05', 'temps_min': 38 + 5/60},
    {'distance': 'Semi-marathon', 'distance_km': 21.0975,'temps_reel': '1:25:19','temps_min': 85 + 19/60},
    {'distance': 'Marathon',      'distance_km': 42.195, 'temps_reel': '2:51:45','temps_min': 171 + 45/60},
]

# Parametres de competition typiques
FC_COMPETITION = 168
DENIVELE = 50
ANNEE = 2026
MOIS = 5
JOUR = 6  # Dimanche

resultats = []
for c in competitions:
    X_pred = pd.DataFrame([{
        'distance_km': c['distance_km'],
        'average_heartrate': FC_COMPETITION,
        'total_elevation_gain': DENIVELE,
        'annee': ANNEE,
        'mois': MOIS,
        'jour_semaine': JOUR
    }])
    allure_pred = modele.predict(X_pred)[0]
    temps_predit_min = allure_pred * c['distance_km']
    allure_reelle = c['temps_min'] / c['distance_km']

    def min_to_str(m):
        h = int(m // 60)
        mins = int(m % 60)
        secs = int((m % 1) * 60)
        if h > 0:
            return f"{h}h{mins:02d}:{secs:02d}"
        return f"{mins}:{secs:02d}"

    resultats.append({
        'Distance': c['distance'],
        'Temps reel': c['temps_reel'],
        'Temps predit': min_to_str(temps_predit_min),
        'Ecart (min)': round(temps_predit_min - c['temps_min'], 1),
        'Allure reelle': f"{int(allure_reelle)}:{int((allure_reelle%1)*60):02d} min/km",
        'Allure predite': f"{int(allure_pred)}:{int((allure_pred%1)*60):02d} min/km",
        'temps_reel_min': c['temps_min'],
        'temps_predit_min': temps_predit_min,
    })

df_res = pd.DataFrame(resultats)

# Metriques
st.header('Resultats')
cols = st.columns(4)
for i, row in df_res.iterrows():
    ecart = row['Ecart (min)']
    signe = '+' if ecart > 0 else ''
    cols[i].metric(
        row['Distance'],
        f"Reel : {row['Temps reel']}",
        f"{signe}{ecart:.1f} min vs prediction"
    )

st.markdown('---')

# Graphique comparaison
st.header('Comparaison visuelle')
fig = go.Figure()
fig.add_trace(go.Bar(
    name='Temps reel',
    x=df_res['Distance'],
    y=df_res['temps_reel_min'],
    marker_color='#FC4C02',
    text=df_res['Temps reel'],
    textposition='outside'
))
fig.add_trace(go.Bar(
    name='Temps predit (ML)',
    x=df_res['Distance'],
    y=df_res['temps_predit_min'],
    marker_color='steelblue',
    text=df_res['Temps predit'],
    textposition='outside'
))
fig.update_layout(
    barmode='group',
    title='Temps reel vs Prediction ML par distance',
    yaxis_title='Temps (minutes)',
    height=450,
    plot_bgcolor='white'
)
st.plotly_chart(fig, use_container_width=True)

# Graphique ecart
st.header('Ecart prediction vs realite')
couleurs = ['green' if e < 0 else 'orangered' for e in df_res['Ecart (min)']]
fig2 = go.Figure(go.Bar(
    x=df_res['Distance'],
    y=df_res['Ecart (min)'],
    marker_color=couleurs,
    text=[f"{'+' if e > 0 else ''}{e:.1f} min" for e in df_res['Ecart (min)']],
    textposition='outside'
))
fig2.add_hline(y=0, line_dash='dash', line_color='black')
fig2.update_layout(
    title='Ecart entre prediction ML et temps reel (+ = modele trop pessimiste)',
    yaxis_title='Ecart (minutes)',
    height=400,
    plot_bgcolor='white'
)
st.plotly_chart(fig2, use_container_width=True)

# Tableau
st.header('Detail par distance')
st.dataframe(
    df_res[['Distance', 'Temps reel', 'Temps predit', 'Allure reelle', 'Allure predite', 'Ecart (min)']],
    use_container_width=True,
    hide_index=True
)

# Insight
st.markdown('---')
st.header('Interpretation')
st.markdown('''
Le modele Random Forest est entraine sur **700+ sorties d entrainement** — des courses faciles,
des intervalles, des longues sorties. Il ne connait pas la notion de competition.

En course officielle, plusieurs facteurs echappent au modele :
- **L adrenaline et la motivation** du jour de course
- **La preparation specifique** (affutage, nutrition, strategie de pace)
- **Les chaussures carbone** (gain estime de 2-4%)
- **L effet peloton** et les conditions meteorologiques

Ces ecarts ne sont pas des erreurs du modele — ils illustrent la limite fondamentale
du machine learning : **un modele ne peut predire que ce qu il a vu**.
L analyste reste indispensable pour interpreter le contexte.
''')
st.caption('Modele Random Forest — R2: 0.813 — entraine sur les donnees Strava personnelles')