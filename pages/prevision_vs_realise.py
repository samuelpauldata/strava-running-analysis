import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title='Prevision vs Realise', layout='wide')
st.title('Prevision ML vs Performances reelles')
st.markdown('Comparaison entre les predictions du modele Random Forest et mes 3 marathons officiels')

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

# Donnees reelles des 3 marathons depuis Strava
marathons = [
    {
        'nom': 'Beneva de Quebec',
        'date': '2024-10-06',
        'annee': 2024,
        'mois': 10,
        'jour_semaine': 6,
        'distance_km': 42.6483,
        'moving_time': 13116,
        'fc': 149.8,
        'denivele': 80,
    },
    {
        'nom': 'Toronto',
        'date': '2025-10-19',
        'annee': 2025,
        'mois': 10,
        'jour_semaine': 6,
        'distance_km': 42.7792,
        'moving_time': 10842,
        'fc': 166.6,
        'denivele': 80,
    },
    {
        'nom': 'Buffalo',
        'date': '2026-05-24',
        'annee': 2026,
        'mois': 5,
        'jour_semaine': 6,
        'distance_km': 42.6245,
        'moving_time': 10419,
        'fc': 167.9,
        'denivele': 50,
    },
]

def secondes_to_str(s):
    h = int(s // 3600)
    m = int((s % 3600) // 60)
    sec = int(s % 60)
    return f"{h}h{m:02d}:{sec:02d}"

def min_to_str(m):
    h = int(m // 60)
    mins = int(m % 60)
    secs = int((m % 1) * 60)
    if h > 0:
        return f"{h}h{mins:02d}:{secs:02d}"
    return f"{mins}:{secs:02d}"

resultats = []
for m in marathons:
    temps_reel_min = m['moving_time'] / 60
    allure_reelle = temps_reel_min / m['distance_km']

    X_pred = pd.DataFrame([{
        'distance_km': m['distance_km'],
        'average_heartrate': m['fc'],
        'total_elevation_gain': m['denivele'],
        'annee': m['annee'],
        'mois': m['mois'],
        'jour_semaine': m['jour_semaine']
    }])
    allure_predite = modele.predict(X_pred)[0]
    temps_predit_min = allure_predite * m['distance_km']
    ecart = temps_predit_min - temps_reel_min

    resultats.append({
        'Marathon': m['nom'],
        'Date': m['date'],
        'Temps reel': secondes_to_str(m['moving_time']),
        'Temps predit': min_to_str(temps_predit_min),
        'Allure reelle': f"{int(allure_reelle)}:{int((allure_reelle%1)*60):02d} min/km",
        'Allure predite': f"{int(allure_predite)}:{int((allure_predite%1)*60):02d} min/km",
        'FC reelle': f"{m['fc']:.0f} bpm",
        'Ecart (min)': round(ecart, 1),
        'temps_reel_min': temps_reel_min,
        'temps_predit_min': temps_predit_min,
    })

df_res = pd.DataFrame(resultats)

# ---- PROGRESSION ----
st.header('Progression sur 3 marathons')
col1, col2, col3 = st.columns(3)
amelioration_total = marathons[0]['moving_time'] - marathons[2]['moving_time']
amelioration_toronto = marathons[0]['moving_time'] - marathons[1]['moving_time']
amelioration_buffalo = marathons[1]['moving_time'] - marathons[2]['moving_time']

col1.metric('Beneva de Quebec 2024', secondes_to_str(marathons[0]['moving_time']))
col2.metric('Toronto 2025', secondes_to_str(marathons[1]['moving_time']),
            f"-{secondes_to_str(amelioration_toronto)} vs Beneva")
col3.metric('Buffalo 2026', secondes_to_str(marathons[2]['moving_time']),
            f"-{secondes_to_str(amelioration_buffalo)} vs Toronto")

st.markdown('---')

# Graphique progression
fig_prog = go.Figure()
fig_prog.add_trace(go.Scatter(
    x=[m['nom'] for m in marathons],
    y=[m['moving_time'] / 60 for m in marathons],
    mode='lines+markers+text',
    text=[secondes_to_str(m['moving_time']) for m in marathons],
    textposition='top center',
    line=dict(color='#FC4C02', width=3),
    marker=dict(size=12, color='#FC4C02'),
    name='Temps reel'
))
fig_prog.update_layout(
    title='Progression des temps de marathon',
    yaxis_title='Temps (minutes)',
    yaxis_autorange='reversed',
    height=400,
    plot_bgcolor='white'
)
st.plotly_chart(fig_prog, use_container_width=True)

st.markdown('---')

# ---- COMPARAISON ML ----
st.header('Prediction ML vs Temps reel')
st.markdown('Le modele est alimenté avec la **vraie FC de chaque course** pour une comparaison juste.')

fig_comp = go.Figure()
fig_comp.add_trace(go.Bar(
    name='Temps reel',
    x=df_res['Marathon'],
    y=df_res['temps_reel_min'],
    marker_color='#FC4C02',
    text=df_res['Temps reel'],
    textposition='outside'
))
fig_comp.add_trace(go.Bar(
    name='Temps predit (ML)',
    x=df_res['Marathon'],
    y=df_res['temps_predit_min'],
    marker_color='#1a1a2e',
    text=df_res['Temps predit'],
    textposition='outside'
))
fig_comp.update_layout(
    barmode='group',
    title='Temps reel vs Prediction ML',
    yaxis_title='Temps (minutes)',
    height=450,
    plot_bgcolor='white'
)
st.plotly_chart(fig_comp, use_container_width=True)

# Graphique ecart
couleurs = ['#2ecc71' if e < 0 else '#FC4C02' for e in df_res['Ecart (min)']]
fig_ecart = go.Figure(go.Bar(
    x=df_res['Marathon'],
    y=df_res['Ecart (min)'],
    marker_color=couleurs,
    text=[f"{'+' if e > 0 else ''}{e:.1f} min" for e in df_res['Ecart (min)']],
    textposition='outside'
))
fig_ecart.add_hline(y=0, line_dash='dash', line_color='black')
fig_ecart.update_layout(
    title='Ecart ML vs reel (+ = modele trop pessimiste, - = modele trop optimiste)',
    yaxis_title='Ecart (minutes)',
    height=400,
    plot_bgcolor='white'
)
st.plotly_chart(fig_ecart, use_container_width=True)

st.markdown('---')

# Tableau
st.header('Detail par marathon')
st.dataframe(
    df_res[['Marathon', 'Date', 'Temps reel', 'Temps predit', 'FC reelle', 'Allure reelle', 'Allure predite', 'Ecart (min)']],
    use_container_width=True,
    hide_index=True
)

st.markdown('---')

# Interpretation
st.header('Interpretation')
st.markdown(f'''
Le modele Random Forest est entraine sur **700+ sorties d entrainement**. En lui fournissant
la **vraie frequence cardiaque** de chaque course, on obtient une comparaison plus juste.

**Ce que les ecarts revelent :**
- **Beneva 2024** : premier marathon, le modele etait {'pessimiste' if df_res.iloc[0]['Ecart (min)'] > 0 else 'optimiste'} de {abs(df_res.iloc[0]['Ecart (min)']):.1f} min
- **Toronto 2025** : avec une FC de 166 bpm, le modele etait {'pessimiste' if df_res.iloc[1]['Ecart (min)'] > 0 else 'optimiste'} de {abs(df_res.iloc[1]['Ecart (min)']):.1f} min  
- **Buffalo 2026** : avec une FC de 168 bpm, le modele etait {'pessimiste' if df_res.iloc[2]['Ecart (min)'] > 0 else 'optimiste'} de {abs(df_res.iloc[2]['Ecart (min)']):.1f} min

**Amelioration totale sur 2 ans : -{secondes_to_str(amelioration_total)}** entre Beneva 2024 et Buffalo 2026.

Le modele ne peut pas capturer l evolution de la forme physique, la preparation specifique,
ni la motivation du jour de course. **L analyste reste indispensable pour interpreter le contexte.**
''')
st.caption('Modele Random Forest — R2: 0.813 — entraine sur les donnees Strava personnelles')