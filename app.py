import streamlit as st
import itertools
import numpy as np
import pandas as pd
import random
import io

# ==========================================
# MOTEUR 1 : FILTRAGE EMPIRIQUE (VBA -> PYTHON)
# ==========================================

def preparer_pool(base_initiale, exclure_10=False, exclure_20=False):
    pool = set()
    for val in base_initiale:
        if val - 1 > 0: pool.add(val - 1)
        pool.add(val + 1)
        
    if exclure_10: pool = {x for x in pool if not (10 <= x <= 19)}
    if exclure_20: pool = {x for x in pool if not (20 <= x <= 25)} # Ajusté à 25 max
        
    return sorted(list(pool))

def generer_combinaisons(pool, base_initiale, forcer_base=True, min_sum=80, max_sum=180, cible_10=-1, cible_20=-1):
    resultats = []
    base_set = set(base_initiale) if base_initiale else set()
    
    for combi in itertools.combinations(pool, 10):
        # 1. Filtre de la somme
        somme_combi = sum(combi)
        if somme_combi < min_sum or (max_sum > 0 and somme_combi > max_sum):
            continue
            
        # 2. Filtre de la base
        if forcer_base and len(set(combi).intersection(base_set)) < 3:
            continue
            
        # 3. Filtre strict des Dizaines (10 à 19)
        if cible_10 != -1:
            nb_10 = sum(1 for x in combi if 10 <= x <= 19)
            if nb_10 != cible_10:
                continue
                
        # 4. Filtre strict des Vingtaines (20 à 25)
        if cible_20 != -1:
            nb_20 = sum(1 for x in combi if 20 <= x <= 25)
            if nb_20 != cible_20:
                continue
                
        resultats.append(combi)
    return resultats

def filtrer_par_historique(combinaisons, historiques, seuil_exclusion=6, max_numero=25):
    if not combinaisons or not historiques:
        return combinaisons
    
    # Création des matrices pour produit vectoriel ultra-rapide
    C = np.zeros((len(combinaisons), max_numero + 1), dtype=np.int8)
    H = np.zeros((len(historiques), max_numero + 1), dtype=np.int8)
    
    for i, combi in enumerate(combinaisons): C[i, list(combi)] = 1
    for i, hist in enumerate(historiques): H[i, list(hist)] = 1
        
    intersections = np.dot(C, H.T)
    max_correspondances = np.max(intersections, axis=1)
    
    masque_valide = max_correspondances <= seuil_exclusion
    combi_array = np.array(combinaisons)
    return [tuple(c) for c in combi_array[masque_valide]]
# ==========================================
# MOTEUR 2 : SYSTEME REDUCTEUR GLOUTON
# ==========================================

def generer_systeme_reducteur(pool_numeros, taille_grille=10, garantie_visee=7, max_iterations=50000):
    cibles_a_couvrir = set(itertools.combinations(pool_numeros, garantie_visee))
    total_cibles = len(cibles_a_couvrir)
    grilles_retenues = []
    iterations = 0
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    while cibles_a_couvrir and iterations < max_iterations:
        iterations += 1
        candidat = tuple(sorted(random.sample(pool_numeros, taille_grille)))
        cibles_du_candidat = set(itertools.combinations(candidat, garantie_visee))
        nouvelles_couvertures = cibles_a_couvrir.intersection(cibles_du_candidat)
        
        if nouvelles_couvertures:
            grilles_retenues.append(candidat)
            cibles_a_couvrir.difference_update(nouvelles_couvertures)
            
        if iterations % 500 == 0:
            couvertes = total_cibles - len(cibles_a_couvrir)
            progress_bar.progress(min(couvertes / total_cibles, 1.0))
            status_text.text(f"Progression : {couvertes}/{total_cibles} cibles couvertes. Grilles : {len(grilles_retenues)}")
            
    progress_bar.progress(1.0)
    status_text.text(f"Terminé. {len(grilles_retenues)} grilles générées. Cibles restantes: {len(cibles_a_couvrir)}")
    return grilles_retenues

def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Grilles')
    return output.getvalue()
# ==========================================
# INTERFACE UTILISATEUR (STREAMLIT)
# ==========================================
st.set_page_config(page_title="Générateur de Grilles (1 à 25)", layout="wide")
st.title("Générateur de Grilles - Analyse Comparative")

# Configuration fixe du pool : 1 à 25
pool_global = list(range(1, 26))
st.sidebar.header("Paramètres Globaux")
st.sidebar.success("Pool verrouillé : Numéros de 1 à 25.")

tab1, tab2 = st.tabs(["1. Moteur Empirique (Filtres Historiques)", "2. Moteur Mathématique (Système Réducteur)"])

with tab1:
    st.header("Filtrage par Hypothèses et Limites")
    col1, col2, col3 = st.columns(3) # On passe sur 3 colonnes pour l'ergonomie
    
    with col1:
        base_input = st.text_input("Série de base :", "5 12 18 21")
        forcer_base = st.checkbox("Forcer min 3 numéros de la base", value=True)
        
    with col2:
        min_sum = st.number_input("Plancher de somme", value=80, step=10)
        max_sum = st.number_input("Plafond de somme", value=180, step=10)
        
    with col3:
        st.markdown("**Structure (Mettre -1 pour ignorer)**")
        cible_10 = st.number_input("Nombre exact de dizaines (10-19)", value=-1, min_value=-1, max_value=10)
        cible_20 = st.number_input("Nombre exact de vingtaines (20-25)", value=-1, min_value=-1, max_value=6)
        seuil_exclu = st.number_input("Seuil d'exclusion historique", value=6, min_value=4, max_value=10)
        
    fichier_historique = st.file_uploader("Importer l'historique des tirages", type=['csv', 'xlsx'])
    
    if st.button("Lancer le Moteur Empirique", type="primary"):
        base_liste = [int(x) for x in base_input.split() if x.isdigit()]
        
        # Le pool ne prend plus les paramètres d'exclusion
        sous_pool = preparer_pool(base_liste)
        pool_final = sorted(list(set(sous_pool).intersection(set(pool_global))))
        if not pool_final: pool_final = pool_global
            
        st.info(f"Pool final de travail : {pool_final} ({len(pool_final)} numéros)")
        
        with st.spinner("Génération et application de la structure spatiale..."):
            # On envoie les nouvelles cibles à la fonction
            grilles_brutes = generer_combinaisons(pool_final, base_liste, forcer_base, min_sum, max_sum, cible_10, cible_20)
        
        st.success(f"Étape 1 : {len(grilles_brutes)} grilles survivent au filtre de structure.")
        
        # --- (Le reste du code pour l'historique et l'export reste identique) ---
        historiques = []
        if fichier_historique:
            df_hist = pd.read_csv(fichier_historique) if fichier_historique.name.endswith('.csv') else pd.read_excel(fichier_historique)
            df_propre = df_hist.iloc[:, :10].dropna()
            historiques = df_propre.astype(int).values.tolist()
        
        if historiques and grilles_brutes:
            grilles_finales = filtrer_par_historique(grilles_brutes, historiques, seuil_exclu)
            st.success(f"Étape 2 : {len(grilles_finales)} grilles retenues après historique.")
        else:
            grilles_finales = grilles_brutes
            
        if grilles_finales:
            df_results = pd.DataFrame(grilles_finales, columns=[f"N{i+1}" for i in range(10)])
            st.dataframe(df_results)
            st.download_button("Télécharger les grilles (Excel)", convert_df_to_excel(df_results), 'grilles_empiriques.xlsx')

with tab2:
    st.header("Système Réducteur (Couverture Géométrique)")
    st.warning("Ce moteur ignore volontairement tout historique et toute somme.")
    
    garantie = st.slider("Garantie visée (Si les 10 gagnants sont dans mon pool de 25, je veux au moins une grille à X bons numéros) :", min_value=5, max_value=9, value=7)
    max_iter = st.number_input("Nombre d'itérations", value=20000, step=5000)
    
    if st.button("Lancer le Calcul Réducteur", type="primary"):
        with st.spinner("Calcul des matrices de couverture..."):
            grilles_reductrices = generer_systeme_reducteur(pool_global, 10, garantie, max_iter)
        
        df_reducteur = pd.DataFrame(grilles_reductrices, columns=[f"N{i+1}" for i in range(10)])
        st.success(f"Système généré ! Total : {len(df_reducteur)} grilles.")
        st.dataframe(df_reducteur)
        st.download_button("Télécharger le système (Excel)", convert_df_to_excel(df_reducteur), 'systeme_reducteur.xlsx')