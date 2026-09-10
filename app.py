import streamlit as st
import pandas as pd
import numpy as np
import itertools
import io

# ==============================================================================
# CONFIGURATION ET PRÉREQUIS
# ==============================================================================
st.set_page_config(page_title="Générateur & Système Réducteur", layout="wide")

# Initialisation de la mémoire persistante pour l'audit
if "grilles_actives" not in st.session_state:
    st.session_state.grilles_actives = None

pool_global = list(range(1, 26))

# ==============================================================================
# FONCTIONS LOGIQUES ET ALGORITHMES
# ==============================================================================

def preparer_pool(base_initiale):
    """
    Génère le pool de travail en prenant strictement les voisins immédiats (+1, -1)
    de chaque numéro de la base, sans conserver la base sauf par chevauchement.
    """
    pool = set()
    for val in base_initiale:
        if val - 1 >= 1:
            pool.add(val - 1)
        if val + 1 <= 25:
            pool.add(val + 1)
    return sorted(list(pool))


def generer_combinaisons(pool, base_initiale, forcer_base=True, min_sum=80, max_sum=180, cible_10=-1, cible_20=-1):
    """
    Moteur combinatoire appliquant les contraintes de somme, d'ancrage et de structure.
    """
    resultats = []
    base_set = set(base_initiale) if base_initiale else set()
    
    for combi in itertools.combinations(pool, 10):
        # 1. Filtre sur la somme
        somme_combi = sum(combi)
        if somme_combi < min_sum or (max_sum > 0 and somme_combi > max_sum):
            continue
            
        # 2. Filtre d'ancrage sur la base
        if forcer_base and len(set(combi).intersection(base_set)) < 3:
            continue
            
        # 3. Filtre topologique : dizaines (10 à 19)
        if cible_10 != -1:
            nb_10 = sum(1 for x in combi if 10 <= x <= 19)
            if nb_10 != cible_10:
                continue
                
        # 4. Filtre topologique : vingtaines (20 à 25)
        if cible_20 != -1:
            nb_20 = sum(1 for x in combi if 20 <= x <= 25)
            if nb_20 != cible_20:
                continue
                
        resultats.append(combi)
    return resultats


def filtrer_par_historique(grilles, historiques, seuil_exclusion=6):
    """
    Suppression matricielle accélérée via Numpy des grilles dépassant le seuil de similitude.
    """
    if not historiques or not grilles:
        return grilles
    
    # Matrice binaire de l'historique (N_lignes x 26)
    H = np.zeros((len(historiques), 26), dtype=np.int8)
    for i, hist in enumerate(historiques):
        H[i, list(hist)] = 1
        
    # Matrice binaire des grilles candidates (M_lignes x 26)
    G = np.zeros((len(grilles), 26), dtype=np.int8)
    for j, g in enumerate(grilles):
        G[j, list(g)] = 1
        
    # Produit matriciel : calcul vectoriel du nombre d'intersections communes
    intersections = np.dot(G, H.T)
    max_communs = np.max(intersections, axis=1)
    
    # Conservation des grilles strictement inférieures ou égales au seuil
    index_valides = np.where(max_communs <= seuil_exclusion)[0]
    return [grilles[idx] for idx in index_valides]


def algorithme_glouton_reducteur(pool, taille_grille=10, garantie=7):
    """
    Algorithme de couverture glouton (Greedy Set Cover)
    Construit un système réduit couvrant les sous-ensembles requis.
    """
    toutes_combinaisons = list(itertools.combinations(pool, taille_grille))
    if len(pool) <= 15:
        sous_ensembles_a_couvrir = set(itertools.combinations(pool, garantie))
    else:
        # Échantillonnage représentatif si l'espace combinatoire sature la mémoire
        sous_ensembles_a_couvrir = set(list(itertools.combinations(pool, garantie))[:15000])

    grilles_retenues = []
    
    while sous_ensembles_a_couvrir:
        meilleure_grille = None
        max_couverts = set()
        
        # Recherche locale de la grille à couverture marginale maximale
        for combi in toutes_combinaisons:
            couverts = set(itertools.combinations(combi, garantie)).intersection(sous_ensembles_a_couvrir)
            if len(couverts) > len(max_couverts):
                max_couverts = couverts
                meilleure_grille = combi
                
        if not meilleure_grille or len(max_couverts) == 0:
            break
            
        grilles_retenues.append(meilleure_grille)
        sous_ensembles_a_couvrir -= max_couverts
        toutes_combinaisons.remove(meilleure_grille)
        
    return grilles_retenues


def evaluer_gains(grilles, tirage_test):
    """
    Mesure le nombre d'intersections entre les grilles en mémoire et un tirage cible.
    """
    set_test = set(tirage_test)
    scores = [len(set(g).intersection(set_test)) for g in grilles]
    return {k: scores.count(k) for k in range(6, 11)}


def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Grilles')
    return output.getvalue()

# ==============================================================================
# INTERFACE UTILISATEUR
# ==============================================================================

st.title("Système de Génération & Réduction Mathématique")
tab1, tab2 = st.tabs(["Moteur 1 : Filtrage Empirique", "Moteur 2 : Système Réducteur Mathématique"])

# ------------------------------------------------------------------------------
# ONGLET 1 : FILTRAGE EMPIRIQUE
# ------------------------------------------------------------------------------
with tab1:
    st.header("Filtrage par Hypothèses et Limites")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        base_input = st.text_input("Série de base :", "3 7 11 15 18 19 22")
        forcer_base = st.checkbox("Forcer min 3 numéros de la base", value=True)
        
    with col2:
        min_sum = st.number_input("Plancher de somme", value=80, step=10)
        max_sum = st.number_input("Plafond de somme", value=180, step=10)
        
    with col3:
        st.markdown("**Structure (-1 pour désactiver)**")
        cible_10 = st.number_input("Nombre exact de dizaines (10-19)", value=-1, min_value=-1, max_value=10)
        cible_20 = st.number_input("Nombre exact de vingtaines (20-25)", value=-1, min_value=-1, max_value=6)
        seuil_exclu = st.number_input("Seuil d'exclusion historique", value=6, min_value=4, max_value=10)
        
    fichier_historique = st.file_uploader("Importer l'historique des tirages", type=['csv', 'xlsx'])
    
    if st.button("Lancer le Moteur Empirique", type="primary"):
        base_liste = [int(x) for x in base_input.split() if x.isdigit()]
        
        sous_pool = preparer_pool(base_liste)
        pool_final = sorted(list(set(sous_pool).intersection(set(pool_global))))
        if not pool_final:
            pool_final = pool_global
            
        st.info(f"Pool final de travail : {pool_final} ({len(pool_final)} numéros)")
        
        with st.spinner("Génération et application des contraintes..."):
            grilles_brutes = generer_combinaisons(pool_final, base_liste, forcer_base, min_sum, max_sum, cible_10, cible_20)
        
        st.write(f"**Étape 1 :** {len(grilles_brutes)} grilles survivent au filtre de structure.")
        
        historiques = []
        if fichier_historique:
            df_hist = pd.read_csv(fichier_historique) if fichier_historique.name.endswith('.csv') else pd.read_excel(fichier_historique)
            df_propre = df_hist.iloc[:, :10].dropna()
            historiques = df_propre.astype(int).values.tolist()
        
        if historiques and grilles_brutes:
            grilles_finales = filtrer_par_historique(grilles_brutes, historiques, seuil_exclu)
            st.success(f"**Étape 2 :** {len(grilles_finales)} grilles retenues après historique.")
        else:
            grilles_finales = grilles_brutes
            
        if grilles_finales:
            # Enregistrement dans la mémoire de session globale
            st.session_state.grilles_actives = grilles_finales
            
            df_results = pd.DataFrame(grilles_finales, columns=[f"N{i+1}" for i in range(10)])
            st.dataframe(df_results)
            st.download_button("Télécharger les grilles (Excel)", convert_df_to_excel(df_results), 'grilles_empiriques.xlsx')
        else:
            st.warning("Aucune grille ne respecte l'ensemble de ces contraintes.")

# ------------------------------------------------------------------------------
# ONGLET 2 : SYSTÈME RÉDUCTEUR MATHÉMATIQUE
# ------------------------------------------------------------------------------
with tab2:
    st.header("Couverture Combinatoire Optimale (Garantie de Rang)")
    
    col_r1, col_r2 = st.columns(2)
    with col_r1:
        pool_reducteur_input = st.text_input("Numéros du pool à couvrir :", "2 4 6 8 10 12 14 16 18 20 22 24")
    with col_r2:
        garantie_cible = st.selectbox("Garantie mathématique visée", [6, 7, 8], index=1)
        
    if st.button("Lancer le Calcul Réducteur", type="primary"):
        pool_r = [int(x) for x in pool_reducteur_input.split() if x.isdigit()]
        pool_r = sorted(list(set(pool_r)))
        
        if len(pool_r) < 10:
            st.error("Le pool doit contenir au minimum 10 numéros.")
        else:
            with st.spinner("Optimisation matricielle de couverture..."):
                grilles_reductrices = algorithme_glouton_reducteur(pool_r, taille_grille=10, garantie=garantie_cible)
            
            st.session_state.grilles_actives = grilles_reductrices
            st.success(f"Système réduit généré : {len(grilles_reductrices)} grilles garantissent un minimum de {garantie_cible}/10.")
            
            df_red = pd.DataFrame(grilles_reductrices, columns=[f"N{i+1}" for i in range(10)])
            st.dataframe(df_red)
            st.download_button("Télécharger le système réduit (Excel)", convert_df_to_excel(df_red), 'systeme_reduit.xlsx')

# ==============================================================================
# MODULE DE VÉRIFICATION A POSTERIORI (AUDIT DU TIRAGE)
# ==============================================================================
st.divider()
st.subheader("Audit & Vérification des Grilles en Mémoire")

if st.session_state.grilles_actives:
    nb_grilles = len(st.session_state.grilles_actives)
    st.info(f"**Jeu chargé :** {nb_grilles} grilles prêtes pour l'audit.")
    
    col_in, col_btn = st.columns([3, 1])
    with col_in:
        tirage_test_input = st.text_input("Saisis les 10 numéros du tirage cible :", "")
    with col_btn:
        st.write("")
        st.write("")
        lancer_audit = st.button("Vérifier les gains", type="secondary")
        
    if lancer_audit:
        tirage_test = [int(x) for x in tirage_test_input.split() if x.isdigit()]
        
        # Validation d'intégrité
        if len(tirage_test) != 10:
            st.error("Erreur de format : saisis exactement 10 numéros.")
        elif len(set(tirage_test)) != 10:
            st.error("Erreur logique : la combinaison ne doit comporter aucun doublon.")
        elif any(x < 1 or x > 25 for x in tirage_test):
            st.error("Périmètre invalide : tous les numéros doivent être compris entre 1 et 25.")
        else:
            bilan = evaluer_gains(st.session_state.grilles_actives, tirage_test)
            
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("6 Bons", bilan.get(6, 0))
            m2.metric("7 Bons", bilan.get(7, 0))
            m3.metric("8 Bons", bilan.get(8, 0))
            m4.metric("9 Bons", bilan.get(9, 0))
            m5.metric("10/10", bilan.get(10, 0))
            
            total_primes = sum(bilan.values())
            ratio = (total_primes / nb_grilles) * 100
            st.write(f"**Total grilles primées (≥ 6 numéros) :** {total_primes} / {nb_grilles} ({ratio:.2f}%)")
else:
    st.caption("Génère d'abord des grilles dans l'un des deux onglets ci-dessus pour activer ce module.")
