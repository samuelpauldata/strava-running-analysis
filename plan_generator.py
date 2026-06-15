import streamlit as st
from dotenv import load_dotenv
load_dotenv('C:/Projets_Data/.env')
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
    # Calcul automatique selon objectif et semaines
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

# Exemples similaires
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
    exemples_texte += f'Semaine {code} ({vol:.0f} km):\n'
    for _, row in rows.iterrows():
        exemples_texte += f"{row['notes']}\n"
    exemples_texte += '\n'

if st.button('Generer mon plan', type='primary'):
    with st.spinner('Generation en cours...'):
        groq_key = os.getenv('GROQ_API_KEY')
        if not groq_key:
            st.error('Cle GROQ manquante!')
        else:
            client = Groq(api_key=groq_key)
            prompt = f'''Tu es un coach de course a pied expert quebecois. Genere un plan d entrainement hebdomadaire structure en Markdown.

PROFIL ACTUEL (donnees Strava reelles) :
- Allure recente : {allure_str}
- Volume hebdo recent : {volume_recent:.1f} km
- Distance max recente : {distance_max:.1f} km
- FC moyenne : {fc_moyenne:.0f} bpm

OBJECTIF :
- Course : {objectif}
- Semaines avant course : {semaines_avant}
- Volume cible : {volume_cible} km

EXEMPLES DU COACH (style a reproduire) :
{exemples_texte}

FORMAT OBLIGATOIRE - utilise exactement ce format Markdown :

## Semaine d entrainement - {objectif} ({volume_cible} km)

| Jour | Type | Details | Distance | Allure |
|------|------|---------|----------|--------|
| Lundi | ... | ... | ... km | ... min/km |
| Mardi | ... | ... | ... km | ... min/km |
| Mercredi | ... | ... | ... km | ... min/km |
| Jeudi | ... | ... | ... km | ... min/km |
| Vendredi | ... | ... | ... km | ... min/km |
| Samedi | ... | ... | ... km | ... min/km |
| Dimanche | ... | ... | ... km | ... min/km |

**Volume total : X km**

### Notes du coach
- Conseil 1
- Conseil 2

REGLES IMPORTANTES :
- Colonne Details : description courte de la seance seulement
- Colonne Distance : uniquement le nombre de km (ex: 14 km)
- Colonne Allure : uniquement l allure (ex: 5:10 min/km)
- Pour les intervalles : mettre la structure dans Details (ex: 10 x 3 min a 3m35)
- Notes du coach : conseils specifiques pour cette semaine selon le profil de l athlete
- Ne jamais melanger les informations entre les colonnes

Genere le plan maintenant :'''
            response = client.chat.completions.create(
                model='llama-3.3-70b-versatile',
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=1000,
                temperature=0.7
            )
            plan = response.choices[0].message.content
            st.header('Votre plan genere')
            st.markdown(plan)
            st.caption(f'Plan genere pour {objectif} - {semaines_avant} semaines avant course - {volume_cible} km cibles')

st.markdown('---')
st.caption('Powered by Groq Llama 3 + vos donnees Strava + style de votre coach')
