import io
import itertools
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# CONFIGURATION ET PRÉREQUIS
# ==============================================================================
st.set_page_config(page_title="Générateur & Système Réducteur", layout="wide")

# Initialisation de la mémoire persistante pour l'audit
if "grilles_actives" not in st.session_state:
    st.session_state.grilles_actives = None

pool_global = list(range(1, 26))

# Limite technique stricte du format Excel (.xlsx)
MAX_EXCEL_ROWS = 1048500

# ==============================================================================
# FONCTIONS LOGIQUES ET ALGORITHMES
# ==============================================================================


def preparer_pool(base_initiale):
    """Génère le pool de travail en prenant strictement les voisins immédiats (+1, -1)

    de chaque numéro de la base, sans conserver la base sauf par chevauchement.
    """
    pool = set()
    for val in base_initiale:
        if val - 1 >= 1:
            pool.add(val - 1)
        if val + 1 <= 25:
            pool.add(val + 1)
    return sorted(list(pool))


def verifier_contraintes_mandel(combi, min_g1=2, min_g2=2):
    """Vérifie les contraintes structurelles de Mandel :

    - Au moins min_g1 numéros dans [1-9]
    - Au moins min_g2 numéros dans [10-19]
    """
    g1 = sum(1 for x in combi if 1 <= x <= 9)
    g2 = sum(1 for x in combi if 10 <= x <= 19)
    return (g1 >= min_g1) and (g2 >= min_g2)


def generer_combinaisons(
    pool,
    base_initiale,
    forcer_base=True,
    min_sum=80,
    max_sum=180,
    cible_10=-1,
    cible_20=-1,
    filtrer_mandel=True,
):
    """Moteur combinatoire appliquant les contraintes de somme, d'ancrage,

    de structure décadaire et le filtre Mandel (>=2 dans [1-9] et >=2 dans
    [10-19]).
    """
    resultats = []
    base_set = set(base_initiale) if base_initiale else set()

    for combi in itertools.combinations(pool, 10):
        # 1. Filtre Mandel (Topologie décadaire minimale)
        if filtrer_mandel and not verifier_contraintes_mandel(
            combi, min_g1=2, min_g2=2
        ):
            continue

        # 2. Filtre sur la somme
        somme_combi = sum(combi)
        if somme_combi < min_sum or (max_sum > 0 and somme_combi > max_sum):
            continue

        # 3. Filtre d'ancrage sur la base
        if forcer_base and len(set(combi).intersection(base_set)) < 3:
            continue

        # 4. Filtre topologique : dizaines cibles (10 à 19)
        if cible_10 != -1:
            nb_10 = sum(1 for x in combi if 10 <= x <= 19)
            if nb_10 != cible_10:
                continue

        # 5. Filtre topologique : vingtaines cibles (20 à 25)
        if cible_20 != -1:
            nb_20 = sum(1 for x in combi if 20 <= x <= 25)
            if nb_20 != cible_20:
                continue

        resultats.append(combi)
    return resultats


def filtrer_par_historique(grilles, historiques, seuil_exclusion=6):
    """Suppression matricielle accélérée via Numpy des grilles dépassant le seuil de similitude."""
    if not historiques or not grilles:
        return grilles

    H = np.zeros((len(historiques), 26), dtype=np.int8)
    for i, hist in enumerate(historiques):
        H[i, list(hist)] = 1

    G = np.zeros((len(grilles), 26), dtype=np.int8)
    for j, g in enumerate(grilles):
        G[j, list(g)] = 1

    intersections = np.dot(G, H.T)
    max_communs = np.max(intersections, axis=1)

    index_valides = np.where(max_communs <= seuil_exclusion)[0]
    return [grilles[idx] for idx in index_valides]


def algorithme_glouton_reducteur(
    pool, taille_grille=10, garantie=8, filtrer_mandel=True, max_grilles=40
):
    """Algorithme de couverture glouton (Greedy Set Cover / Condensation Mandel)."""
    candidats = list(itertools.combinations(pool, taille_grille))

    if filtrer_mandel:
        candidats = [
            c
            for c in candidats
            if verifier_contraintes_mandel(c, min_g1=2, min_g2=2)
        ]

    if not candidats:
        return []

    tous_subsets = list(itertools.combinations(pool, garantie))
    if len(tous_subsets) > 15000:
        sous_ensembles_a_couvrir = set(tous_subsets[:15000])
    else:
        sous_ensembles_a_couvrir = set(tous_subsets)

    grilles_retenues = []

    while sous_ensembles_a_couvrir and len(grilles_retenues) < max_grilles:
        meilleure_grille = None
        max_couverts = set()

        for combi in candidats:
            couverts = set(
                itertools.combinations(combi, garantie)
            ).intersection(sous_ensembles_a_couvrir)
            if len(couverts) > len(max_couverts):
                max_couverts = couverts
                meilleure_grille = combi

        if not meilleure_grille or len(max_couverts) == 0:
            if candidats and len(grilles_retenues) < 15:
                grilles_retenues.append(candidats.pop(0))
                continue
            break

        grilles_retenues.append(meilleure_grille)
        sous_ensembles_a_couvrir -= max_couverts
        candidats.remove(meilleure_grille)

    return grilles_retenues


def evaluer_gains(grilles, tirage_test):
    """Mesure le nombre d'intersections entre les grilles en mémoire et un tirage cible."""
    set_test = set(tirage_test)
    scores = [len(set(g).intersection(set_test)) for g in grilles]
    return {k: scores.count(k) for k in range(0, 11)}


def convert_df_to_excel(df):
    """Génère l'export Excel en protégeant contre le dépassement des 1 048 576 lignes."""
    output = io.BytesIO()
    df_export = df.iloc[:MAX_EXCEL_ROWS] if len(df) > MAX_EXCEL_ROWS else df
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_export.to_excel(writer, index=False, sheet_name="Grilles")
    return output.getvalue()


def convert_df_to_csv(df):
    """Génère l'export CSV sans aucune contrainte de nombre de lignes."""
    return df.to_csv(index=False).encode("utf-8")


# ==============================================================================
# INTERFACE UTILISATEUR
# ==============================================================================

st.title("Système de Génération & Réduction Mathématique")
tab1, tab2 = st.tabs(
    ["Moteur 1 : Filtrage Empirique", "Moteur 2 : Système Réducteur Mandel"]
)

# ------------------------------------------------------------------------------
# ONGLET 1 : FILTRAGE EMPIRIQUE
# ------------------------------------------------------------------------------
with tab1:
    st.header("Filtrage par Hypothèses et Limites")
    col1, col2, col3 = st.columns(3)

    with col1:
        base_input = st.text_input("Série de base :", "3 7 11 15 18 19 22")
        forcer_base = st.checkbox("Forcer min 3 numéros de la base", value=True)
        filtre_mandel_tab1 = st.checkbox(
            "Filtre Mandel : min 2 dans [1-9] et min 2 dans [10-19]", value=True
        )

    with col2:
        min_sum = st.number_input("Plancher de somme", value=80, step=10)
        max_sum = st.number_input("Plafond de somme", value=180, step=10)

    with col3:
        st.markdown("**Structure (-1 pour désactiver)**")
        cible_10 = st.number_input(
            "Nombre exact de dizaines (10-19)",
            value=-1,
            min_value=-1,
            max_value=10,
        )
        cible_20 = st.number_input(
            "Nombre exact de vingtaines (20-25)",
            value=-1,
            min_value=-1,
            max_value=6,
        )
        seuil_exclu = st.number_input(
            "Seuil d'exclusion historique", value=6, min_value=4, max_value=10
        )

    fichier_historique = st.file_uploader(
        "Importer l'historique des tirages", type=["csv", "xlsx"]
    )

    if st.button("Lancer le Moteur Empirique", type="primary"):
        base_liste = [int(x) for x in base_input.split() if x.isdigit()]

        sous_pool = preparer_pool(base_liste)
        pool_final = sorted(
            list(set(sous_pool).intersection(set(pool_global)))
        )
        if not pool_final:
            pool_final = pool_global

        st.info(
            f"Pool final de travail : {pool_final} ({len(pool_final)} numéros)"
        )

        with st.spinner("Génération et application des contraintes..."):
            grilles_brutes = generer_combinaisons(
                pool_final,
                base_liste,
                forcer_base,
                min_sum,
                max_sum,
                cible_10,
                cible_20,
                filtrer_mandel=filtre_mandel_tab1,
            )

        st.write(
            f"**Étape 1 :** {len(grilles_brutes)} grilles survivent aux filtres combinatoires."
        )

        historiques = []
        if fichier_historique:
            df_hist = (
                pd.read_csv(fichier_historique)
                if fichier_historique.name.endswith(".csv")
                else pd.read_excel(fichier_historique)
            )
            df_propre = df_hist.iloc[:, :10].dropna()
            historiques = df_propre.astype(int).values.tolist()

        if historiques and grilles_brutes:
            grilles_finales = filtrer_par_historique(
                grilles_brutes, historiques, seuil_exclu
            )
            st.success(
                f"**Étape 2 :** {len(grilles_finales)} grilles retenues après historique."
            )
        else:
            grilles_finales = grilles_brutes

        if grilles_finales:
            st.session_state.grilles_actives = grilles_finales

            df_results = pd.DataFrame(
                grilles_finales, columns=[f"N{i+1}" for i in range(10)]
            )
            df_results["G1 [1-9]"] = [
                sum(1 for x in g if 1 <= x <= 9) for g in grilles_finales
            ]
            df_results["G2 [10-19]"] = [
                sum(1 for x in g if 10 <= x <= 19) for g in grilles_finales
            ]

            # Affichage allégé dans l'interface si le volume est colossal
            st.dataframe(df_results.head(10000))
            if len(df_results) > 10000:
                st.caption(
                    f"Affichage limité aux 10 000 premières grilles sur {len(df_results):,} pour préserver la fluidité."
                )

            # Gestion des boutons de téléchargement selon le volume
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.download_button(
                    "Télécharger toutes les grilles (CSV)",
                    convert_df_to_csv(df_results),
                    "grilles_empiriques.csv",
                    "text/csv",
                )

            with col_d2:
                if len(df_results) > MAX_EXCEL_ROWS:
                    st.warning(
                        f"Le volume ({len(df_results):,} lignes) dépasse la limite Excel (1 048 576). Le fichier Excel sera tronqué à 1 048 500 lignes. Privilégie le format CSV."
                    )
                    st.download_button(
                        "Télécharger (Excel tronqué)",
                        convert_df_to_excel(df_results),
                        "grilles_empiriques_tronque.xlsx",
                    )
                else:
                    st.download_button(
                        "Télécharger les grilles (Excel)",
                        convert_df_to_excel(df_results),
                        "grilles_empiriques.xlsx",
                    )
        else:
            st.warning(
                "Aucune grille ne respecte l'ensemble de ces contraintes."
            )

# ------------------------------------------------------------------------------
# ONGLET 2 : SYSTÈME RÉDUCTEUR MANDEL
# ------------------------------------------------------------------------------
with tab2:
    st.header("Couverture Combinatoire Optimale (Garantie de Rang)")

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        pool_reducteur_input = st.text_input(
            "Numéros du pool à couvrir :",
            "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25",
        )
    with col_r2:
        garantie_cible = st.selectbox(
            "Garantie mathématique visée", [6, 7, 8], index=2
        )
    with col_r3:
        max_grilles_slider = st.slider(
            "Nombre max de grilles (budget)",
            min_value=5,
            max_value=60,
            value=20,
        )
        filtre_mandel_tab2 = st.checkbox(
            "Forcer condition : ≥2 [1-9] et ≥2 [10-19]", value=True
        )

    if st.button("Lancer le Calcul Réducteur", type="primary"):
        pool_r = [int(x) for x in pool_reducteur_input.split() if x.isdigit()]
        pool_r = sorted(list(set(pool_r)))

        g1_count = len([x for x in pool_r if 1 <= x <= 9])
        g2_count = len([x for x in pool_r if 10 <= x <= 19])

        if len(pool_r) < 10:
            st.error("Le pool doit contenir au minimum 10 numéros.")
        elif filtre_mandel_tab2 and (g1_count < 2 or g2_count < 2):
            st.error(
                "Le pool saisi ne contient pas assez d'éléments pour respecter la contrainte Mandel (min 2 dans [1-9] et min 2 dans [10-19])."
            )
        else:
            with st.spinner(
                "Optimisation matricielle de condensation Mandel..."
            ):
                grilles_reductrices = algorithme_glouton_reducteur(
                    pool_r,
                    taille_grille=10,
                    garantie=garantie_cible,
                    filtrer_mandel=filtre_mandel_tab2,
                    max_grilles=max_grilles_slider,
                )

            st.session_state.grilles_actives = grilles_reductrices
            st.success(
                f"Système réduit généré : {len(grilles_reductrices)} grilles optimisées pour garantir un rang ≥ {garantie_cible}/10."
            )

            df_red = pd.DataFrame(
                grilles_reductrices, columns=[f"N{i+1}" for i in range(10)]
            )
            df_red["G1 [1-9]"] = [
                sum(1 for x in g if 1 <= x <= 9) for g in grilles_reductrices
            ]
            df_red["G2 [10-19]"] = [
                sum(1 for x in g if 10 <= x <= 19) for g in grilles_reductrices
            ]
            st.dataframe(df_red)

            col_dr1, col_dr2 = st.columns(2)
            with col_dr1:
                st.download_button(
                    "Télécharger (CSV)",
                    convert_df_to_csv(df_red),
                    "systeme_reduit.csv",
                    "text/csv",
                )
            with col_dr2:
                st.download_button(
                    "Télécharger (Excel)",
                    convert_df_to_excel(df_red),
                    "systeme_reduit.xlsx",
                )

# ==============================================================================
# MODULE DE VÉRIFICATION A POSTERIORI (AUDIT DU TIRAGE ET COMPARAISON)
# ==============================================================================
st.divider()
st.subheader("Audit & Vérification des Grilles en Mémoire")

if st.session_state.grilles_actives:
    nb_grilles = len(st.session_state.grilles_actives)
    st.info(f"**Jeu chargé :** {nb_grilles} grilles prêtes pour l'audit.")

    col_in, col_btn = st.columns([3, 1])
    with col_in:
        tirage_test_input = st.text_input(
            "Saisis la bonne combinaison de 10 numéros à comparer :",
            "2 5 11 14 18 20 21 22 24 25",
        )
    with col_btn:
        st.write("")
        st.write("")
        lancer_audit = st.button("Comparer au tirage", type="secondary")

    if lancer_audit:
        tirage_test = [
            int(x) for x in tirage_test_input.split() if x.isdigit()
        ]

        if len(tirage_test) != 10:
            st.error("Erreur de format : saisis exactement 10 numéros.")
        elif len(set(tirage_test)) != 10:
            st.error(
                "Erreur logique : la combinaison ne doit comporter aucun doublon."
            )
        elif any(x < 1 or x > 25 for x in tirage_test):
            st.error(
                "Périmètre invalide : tous les numéros doivent être compris entre 1 et 25."
            )
        else:
            bilan = evaluer_gains(st.session_state.grilles_actives, tirage_test)
            set_gagnante = set(tirage_test)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("6 Bons", bilan.get(6, 0))
            m2.metric("7 Bons", bilan.get(7, 0))
            m3.metric("8 Bons (Objectif)", bilan.get(8, 0))
            m4.metric("9 Bons", bilan.get(9, 0))
            m5.metric("10/10 (Jackpot)", bilan.get(10, 0))

            total_objectif_8 = (
                bilan.get(8, 0) + bilan.get(9, 0) + bilan.get(10, 0)
            )
            if total_objectif_8 > 0:
                st.success(
                    f"🎯 Objectif Mandel atteint ! {total_objectif_8} grille(s) atteignent ou dépassent 8/10."
                )
            else:
                st.warning(
                    "Aucune grille n'atteint le seuil de 8/10 sur ce tirage spécifique."
                )

            st.markdown("#### Détail par grille")
            table_rows = []
            # Pour l'audit visuel, on limite l'affichage aux 500 premières grilles si le volume est géant
            audit_sample = st.session_state.grilles_actives[:500]
            for idx, g in enumerate(audit_sample, 1):
                communs = sorted(list(set(g).intersection(set_gagnante)))
                c1 = sum(1 for x in g if 1 <= x <= 9)
                c2 = sum(1 for x in g if 10 <= x <= 19)
                table_rows.append(
                    {
                        "Grille": f"#{idx:02d}",
                        "Composition": " - ".join(f"{x:02d}" for x in g),
                        "G1 [1-9]": c1,
                        "G2 [10-19]": c2,
                        "Bons Numéros": " - ".join(f"{x:02d}" for x in communs)
                        if communs
                        else "-",
                        "Score": f"{len(communs)} / 10",
                        "Statut": "⭐ GAGNANT"
                        if len(communs) >= 8
                        else ("PRIMÉ" if len(communs) >= 6 else "-"),
                    }
                )

            df_detail = pd.DataFrame(table_rows)
            st.dataframe(df_detail, use_container_width=True)
            if len(st.session_state.grilles_actives) > 500:
                st.caption(
                    f"Audit visuel détaillé sur les 500 premières grilles sur un total de {len(st.session_state.grilles_actives):,} (les métriques en haut couvrent 100% des grilles)."
                )
else:
    st.caption(
        "Génère d'abord des grilles dans l'un des deux onglets ci-dessus pour activer ce module."
    )
