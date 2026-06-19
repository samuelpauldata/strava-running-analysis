import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

st.set_page_config(page_title='Prevision vs Realise', layout='wide')
st.title('Prevision ML vs Performances reelles')
st.markdown('Comparaison entre les predictions du modele Random Forest et mes marathons officiels')

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

# Detection automatique des marathons officiels
# Criteres : distance > 40 km + FC > 155 bpm (effort de competition)
@st.cache_data
def detecter_marathons(df):
    # Marathons detectes automatiquement
    df_auto = df[
        (df['distance_km'] > 40) &
        (df['average_heartrate'] > 155)
    ].copy()

    # Beneva 2024 - exception hardcodee (premier marathon, FC conservative 149.8)
    beneva = df[df['start_date_local'].dt.strftime('%Y-%m-%d') == '2024-10-06'].copy()
    beneva = beneva[beneva['distance_km'] > 40]

    # Combiner et dedupliquer
    df_marathons = pd.concat([beneva, df_auto]).drop_duplicates(subset='start_date_local')
    df_marathons = df_marathons.sort_values('start_date_local').reset_index(drop=True)
    return df_marathons

df_marathons = detecter_marathons(df)

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

# Calcul predictions ML pour chaque marathon
resultats = []
for _, row in df_marathons.iterrows():
    temps_reel_min = row['moving_time'] / 60
    allure_reelle = row['allure_min_km']
    denivele = row['total_elevation_gain'] if pd.notna(row.get('total_elevation_gain')) else 50

    X_pred = pd.DataFrame([{
        'distance_km': row['distance_km'],
        'average_heartrate': row['average_heartrate'],
        'total_elevation_gain': denivele,
        'annee': row['annee'],
        'mois': row['mois'],
        'jour_semaine': row['jour_semaine']
    }])
    allure_predite = modele.predict(X_pred)[0]
    temps_predit_min = allure_predite * row['distance_km']
    ecart = temps_predit_min - temps_reel_min

    resultats.append({
        'Marathon': 'Buffalo Marathon' if row['start_date_local'].strftime('%Y-%m-%d') == '2026-05-24' else row['name'],
        'Date': row['start_date_local'].strftime('%Y-%m-%d'),
        'Annee': row['annee'],
        'Temps reel': secondes_to_str(row['moving_time']),
        'Temps predit': min_to_str(temps_predit_min),
        'Allure reelle': f"{int(allure_reelle)}:{int((allure_reelle%1)*60):02d} min/km",
        'Allure predite': f"{int(allure_predite)}:{int((allure_predite%1)*60):02d} min/km",
        'FC reelle': f"{row['average_heartrate']:.0f} bpm",
        'Ecart (min)': round(ecart, 1),
        'temps_reel_min': temps_reel_min,
        'temps_predit_min': temps_predit_min,
    })

df_res = pd.DataFrame(resultats)

# ---- PROGRESSION ----
st.header(f'Progression sur {len(df_res)} marathons')

cols = st.columns(len(df_res))
for i, (_, row) in enumerate(df_res.iterrows()):
    if i == 0:
        cols[i].metric(f"{row['Marathon']} ({row['Annee']})", row['Temps reel'])
    else:
        prev_time = df_res.iloc[i-1]['temps_reel_min']
        amelioration = prev_time - row['temps_reel_min']
        signe = '-' if amelioration > 0 else '+'
        cols[i].metric(
            f"{row['Marathon']} ({row['Annee']})",
            row['Temps reel'],
            f"{signe}{min_to_str(abs(amelioration))} vs precedent"
        )

st.markdown('---')

# Graphique progression
fig_prog = go.Figure()
fig_prog.add_trace(go.Scatter(
    x=df_res['Marathon'] + ' ' + df_res['Annee'].astype(str),
    y=df_res['temps_reel_min'],
    mode='lines+markers+text',
    text=df_res['Temps reel'],
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
st.markdown('Le modele est alimente avec la **vraie FC de chaque course** pour une comparaison juste.')

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
amelioration_totale = df_res.iloc[0]['temps_reel_min'] - df_res.iloc[-1]['temps_reel_min']
st.header('Interpretation')
st.markdown(f'''
Le modele Random Forest est entraine sur **700+ sorties d entrainement**. En lui fournissant
la **vraie frequence cardiaque** de chaque course, on obtient une comparaison plus juste.

**Amelioration totale : +{min_to_str(abs(amelioration_totale))}** entre {df_res.iloc[0]['Marathon']} et {df_res.iloc[-1]['Marathon']}.

Les marathons futurs seront **detectes automatiquement** dans les donnees Strava
si la distance depasse 40 km et la FC moyenne depasse 155 bpm.

Le modele ne peut pas capturer l evolution de la forme physique, la preparation specifique,
ni la motivation du jour de course. **L analyste reste indispensable pour interpreter le contexte.**
''')
st.caption('Modele Random Forest — R2: 0.813 — entraine sur les donnees Strava personnelles')