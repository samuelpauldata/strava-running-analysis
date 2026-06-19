import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import pandas as pd
import os
from groq import Groq

st.set_page_config(page_title='Generateur de plan', layout='wide')
st.title('Generateur de plan d entrainement')
st.markdown('Plan personnalise base sur vos donnees Strava et le style de votre coach')

@st.cache_data
def charger_donnees():
    df = pd.read_csv('strava_courses.csv')
    df['start_date_local'] = pd.to_datetime(df['start_date_local'])
    df['semaine'] = df['start_date_local'].dt.to_period('W')
    df['duree_min'] = df['moving_time'] / 60
    df['allure_min_km'] = df['duree_min'] / df['distance_km']
    return df

@st.cache_data
def charger_plan():
    return pd.read_csv('entrainement_parsed.csv')

df = charger_donnees()
df_plan = charger_plan()

# Profil actuel
volume_par_semaine = df.groupby('semaine')['distance_km'].sum()
volume_recent = volume_par_semaine.tail(4).mean()
df_recent = df.sort_values('start_date_local').tail(20)
allure_recente = df_recent['allure_min_km'].mean()
distance_max = df_recent['distance_km'].max()
fc_moyenne = df_recent['average_heartrate'].mean()
allure_str = f"{int(allure_recente)}:{int((allure_recente%1)*60):02d} min/km"

# Afficher profil
st.header('Votre profil actuel (Strava)')
col1, col2, col3, col4 = st.columns(4)
col1.metric('Allure recente', allure_str)
col2.metric('Volume hebdo recent', f'{volume_recent:.1f} km')
col3.metric('Distance max recente', f'{distance_max:.1f} km')
col4.metric('FC moyenne', f'{fc_moyenne:.0f} bpm')

st.markdown('---')

# Parametres du plan
st.header('Parametres du plan')
col1, col2, col3 = st.columns(3)
with col1:
    objectif = st.selectbox('Objectif', [
        'Pas d evenement - Maintien forme',
        'Pas d evenement - Amelioration generale',
        '5 km',
        '10 km',
        'Demi-marathon',
        'Marathon'
    ])
with col2:
    if 'Pas d evenement' in objectif:
        semaines_avant = 0
        st.info('Pas de course cible - plan de maintien ou progression generale')
    else:
        semaines_avant = st.slider('Semaines avant la course', 1, 24, 12)
with col3:
    if 'Pas d evenement' in objectif:
        if 'Maintien' in objectif:
            volume_cible = int(volume_recent)
            st.metric('Volume cible calcule', f'{volume_cible} km', 'Maintien')
        else:
            volume_cible = int(volume_recent * 1.05)
            st.metric('Volume cible calcule', f'{volume_cible} km', '+5% progression')
    elif semaines_avant <= 2:
        volume_cible = int(volume_recent * 0.6)
        st.metric('Volume cible calcule', f'{volume_cible} km', 'Semaine de competition')
    elif semaines_avant <= 4:
        volume_cible = int(volume_recent * 0.8)
        st.metric('Volume cible calcule', f'{volume_cible} km', 'Affutage')
    elif semaines_avant <= 8:
        volume_cible = int(volume_recent * 1.05)
        st.metric('Volume cible calcule', f'{volume_cible} km', '+5% charge')
    else:
        volume_cible = int(volume_recent * 1.10)
        st.metric('Volume cible calcule', f'{volume_cible} km', '+10% charge')

# RAG - exemples similaires depuis entrainement_parsed.csv
volume_semaines = df_plan[df_plan['type'] != 'off'].groupby('semaine_code')['distance_km'].sum().reset_index()
volume_semaines.columns = ['semaine_code', 'volume']
semaines_similaires = volume_semaines[
    (volume_semaines['volume'] >= volume_cible - 10) &
    (volume_semaines['volume'] <= volume_cible + 10)
].head(3)['semaine_code'].tolist()

exemples_df = df_plan[df_plan['semaine_code'].isin(semaines_similaires) & (df_plan['type'] != 'off')]
exemples_texte = ''
for code in semaines_similaires:
    vol = volume_semaines[volume_semaines['semaine_code'] == code]['volume'].values[0]
    rows = exemples_df[exemples_df['semaine_code'] == code]
    exemples_texte += f'--- Exemple ({vol:.0f} km) ---\n'
    for _, row in rows.iterrows():
        exemples_texte += f"{row['notes']}\n"
    exemples_texte += '\n'

if not semaines_similaires:
    exemples_texte = '(Aucun exemple similaire trouve dans les donnees)'

if st.button('Generer mon plan', type='primary'):
    with st.spinner('Generation en cours...'):
        groq_key = os.getenv('GROQ_API_KEY')
        if not groq_key:
            st.error('Cle GROQ manquante!')
        else:
            client = Groq(api_key=groq_key)
            prompt = f'''Tu es un coach de course a pied quebecois. Tu generes des plans d entrainement exactement dans TON style d ecriture personnel — pas en Markdown formate, pas en tableau. Juste du texte brut, une seance par ligne, comme tu les envoies a tes athletes.

TON STYLE PERSONNEL (regles absolues) :
- Chaque seance = une seule ligne de texte brut
- Sorties faciles : "[Jour] X km ext a XmXX"
  Exemple : "Mercredi 14 km ext a 5m20"
- Intervalles tapis : "[Jour] tapis roulant | [echauff] a XmXX pause X min / | N x [duree/dist] a XmXX pause X min entre tes [duree/dist] / | [retour calme] a XmXX"
  Exemple : "Vendredi tapis roulant | 10 min a 5m10 pause 2 min / | 9 x 2 min a 4m10 pause 1 min entre tes 2 min / | 10 min a 5m10"
- Intervalles ext : "[Jour] X km ext a XmXX pause X min / | N x [dist] a XmXX pause XmXX entre tes [dist] pas + vite / | X km a XmXX"
  Exemple : "Mardi 2 km ext a 4m40 pause 2 min / | 12 x 400m a 3m30 pause 1m25 entre tes 400m pas + vite / | 2 km a 4m40"
- Sorties longues avec finish : "[Jour] X km ext a XmXX tes N dernier + rapide"
  Exemple : "Dimanche 22 km ext a 5m35 tes 5 dernier + rapide"
- Tempo : "[Jour] X km ext tempo | X km a XmXX | X km a XmXX | X km a XmXX"
- Repos : "[Jour] off"
- JAMAIS de tableau, JAMAIS de Markdown, JAMAIS de bullet points
- Tutoiement toujours (tes, entre tes, pas + vite)
- Allures en format XmXX (ex: 4m20, 5m10) — jamais 4:20
- "ext" pour exterieur, "/" comme separateur dans intervalles

PROFIL STRAVA DE L ATHLETE :
- Allure recente : {allure_str}
- Volume hebdo recent : {volume_recent:.1f} km
- Distance max recente : {distance_max:.1f} km
- FC moyenne : {fc_moyenne:.0f} bpm

OBJECTIF :
- Course : {objectif}
- Semaines avant course : {semaines_avant}
- Volume cible cette semaine : {volume_cible} km

EXEMPLES REELS DE TES PLANS (reproduis ce style exactement) :
{exemples_texte}

REGLES DE STRUCTURE :
- Lundi : toujours off
- Mardi et/ou Vendredi : seances de qualite (intervalles tapis ou ext selon la periode)
- Mercredi et Jeudi : sorties en endurance ext
- Samedi : sortie moderee ou off selon le volume
- Dimanche : longue sortie avec progression finale ("tes X dernier + rapide")
- Respecte le volume cible de {volume_cible} km au total
- Adapte l intensite selon les semaines avant la course ({semaines_avant} semaines)

Genere maintenant la semaine complete (Lundi a Dimanche), une seance par ligne. Commence directement par "Lundi" sans introduction ni conclusion.'''

            response = client.chat.completions.create(
                model='llama-3.3-70b-versatile',
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=1500,
                temperature=0.4
            )
            plan = response.choices[0].message.content
            st.header('Votre plan genere')
            st.text(plan)
            st.caption(f'Plan genere pour {objectif} - {semaines_avant} semaines avant course - {volume_cible} km cibles')

st.markdown('---')
st.caption('Powered by Groq Llama 3 + vos donnees Strava + style de votre coach')