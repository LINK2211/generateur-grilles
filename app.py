import streamlit as st

st.set_page_config(
    page_title="Nom de Ton Application",
    page_icon="🚀",
    layout="centered"  # ou "wide"
)
import io
import itertools
import random
from typing import List
import numpy as np
import pandas as pd
import streamlit as st

# ==============================================================================
# CONFIGURATION ET PRÉREQUIS
# ==============================================================================
st.set_page_config(page_title="Générateur & Système Réducteur", layout="wide")

if "grilles_actives" not in st.session_state:
    st.session_state.grilles_actives = None

pool_global = list(range(1, 26))

# ==============================================================================
# FONCTIONS LOGIQUES ET ALGORITHMES
# ==============================================================================


def preparer_pool(base_initiale):
    """Génère le pool de travail en prenant les voisins (+1, -1) de la base."""
    pool = set()
    for val in base_initiale:
        if val - 1 >= 1:
            pool.add(val - 1)
        if val + 1 <= 25:
            pool.add(val + 1)
    return sorted(list(pool))


def verifier_contraintes_decades(combi, min_g1=2, min_g2=2):
    """Vérifie : au moins min_g1 dans [1-9] et min_g2 dans [10-19]."""
    g1 = sum(1 for x in combi if 1 <= x <= 9)
    g2 = sum(1 for x in combi if 10 <= x <= 19)
    return (g1 >= min_g1) and (g2 >= min_g2)


def generer_grilles_selectives(
    pool: List[int],
    base_initiale: List[int],
    nb_grilles: int = 15,
    forcer_base: bool = True,
    min_sum: int = 80,
    max_sum: int = 180,
    cible_10: int = -1,
    cible_20: int = -1,
    forcer_decades: bool = True,
) -> List[List[int]]:
    """Moteur 1 : génération sélective par échantillonnage rapide sous filtres."""
    g1 = [x for x in pool if 1 <= x <= 9]
    g2 = [x for x in pool if 10 <= x <= 19]
    base_set = set(base_initiale) if base_initiale else set()

    if forcer_decades and (len(g1) < 2 or len(g2) < 2):
        return []

    grilles = []
    seen = set()
    attempts = 0
    max_attempts = 40000

    while len(grilles) < nb_grilles and attempts < max_attempts:
        attempts += 1
        ticket = set()

        if forcer_decades:
            k1 = random.choice([2, 3]) if len(g1) >= 3 else 2
            k2 = random.choice([2, 3]) if len(g2) >= 3 else 2
            ticket.update(random.sample(g1, k1))
            ticket.update(random.sample(g2, k2))

        rest_pool = [x for x in pool if x not in ticket]
        needed = 10 - len(ticket)
        if len(rest_pool) < needed:
            continue
        ticket.update(random.sample(rest_pool, needed))

        combi = sorted(list(ticket))

        somme_combi = sum(combi)
        if somme_combi < min_sum or (max_sum > 0 and somme_combi > max_sum):
            continue

        if forcer_base and len(set(combi).intersection(base_set)) < 3:
            continue

        if cible_10 != -1:
            if sum(1 for x in combi if 10 <= x <= 19) != cible_10:
                continue

        if cible_20 != -1:
            if sum(1 for x in combi if 20 <= x <= 25) != cible_20:
                continue

        ticket_tuple = tuple(combi)
        if ticket_tuple not in seen:
            seen.add(ticket_tuple)
            grilles.append(combi)

    return grilles


def algorithme_glouton_mandel(
    pool: List[int],
    taille_grille: int = 10,
    garantie: int = 8,
    filtrer_decades: bool = True,
    max_grilles: int = 20,
) -> List[List[int]]:
    """Moteur 2 : algorithme glouton réducteur de Mandel (Set Cover).

    Maximise la couverture combinatoire des t-uplets avec quota de grilles.
    """
    candidats = list(itertools.combinations(pool, taille_grille))

    if filtrer_decades:
        candidats = [
            c
            for c in candidats
            if verifier_contraintes_decades(c, min_g1=2, min_g2=2)
        ]

    if not candidats:
        return []

    # Sous-ensembles cibles (échantillonnés si l'espace est saturé)
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
            if candidats and len(grilles_retenues) < max_grilles:
                grilles_retenues.append(candidats.pop(0))
                continue
            break

        grilles_retenues.append(list(meilleure_grille))
        sous_ensembles_a_couvrir -= max_couverts
        candidats.remove(meilleure_grille)

    return grilles_retenues


def filtrer_par_historique(grilles, historiques, seuil_exclusion=6):
    """Suppression matricielle accélérée via Numpy des grilles trop similaires à l'historique."""
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


def convert_df_to_excel(df, sheet_name="Grilles"):
    """Exporte un DataFrame propre en Excel avec colonnes auto-dimensionnées."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
        worksheet = writer.sheets[sheet_name]
        for idx, col in enumerate(df.columns):
            max_len = (
                max(df[col].astype(str).map(len).max(), len(str(col))) + 2
            )
            worksheet.set_column(idx, idx, max_len)
    return output.getvalue()


# ==============================================================================
# INTERFACE UTILISATEUR
# ==============================================================================

st.title("Système de Génération & Réduction Mathématique")
tab1, tab2 = st.tabs(
    ["Moteur 1 : Filtrage Empirique", "Moteur 2 : Système Réducteur Mandel"]
)

# ------------------------------------------------------------------------------
# ONGLET 1 : FILTRAGE EMPIRIQUE (SÉLECTIF)
# ------------------------------------------------------------------------------
with tab1:
    st.header("Filtrage par Hypothèses et Limites")
    col1, col2, col3 = st.columns(3)

    with col1:
        base_input = st.text_input("Série de base :", "3 7 11 15 18 19 22")
        forcer_base = st.checkbox("Forcer min 3 numéros de la base", value=True)
        forcer_decades_t1 = st.checkbox(
            "Forcer min 2 dans [1-9] et min 2 dans [10-19]",
            value=True,
            key="decades_t1",
        )

    with col2:
        min_sum = st.number_input("Plancher de somme", value=80, step=10)
        max_sum = st.number_input("Plafond de somme", value=180, step=10)
        nb_grilles_demande = st.number_input(
            "Nombre de grilles à générer", value=20, min_value=1, max_value=200
        )

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
        "Importer l'historique des tirages (optionnel)",
        type=["csv", "xlsx"],
        key="hist_t1",
    )

    if st.button("Générer les Grilles Sélectives", type="primary"):
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

        with st.spinner("Génération sélective des grilles..."):
            quota_recherche = (
                nb_grilles_demande * 4
                if fichier_historique
                else nb_grilles_demande
            )
            grilles_brutes = generer_grilles_selectives(
                pool=pool_final,
                base_initiale=base_liste,
                nb_grilles=quota_recherche,
                forcer_base=forcer_base,
                min_sum=min_sum,
                max_sum=max_sum,
                cible_10=cible_10,
                cible_20=cible_20,
                forcer_decades=forcer_decades_t1,
            )

        historiques = []
        if fichier_historique and grilles_brutes:
            df_hist = (
                pd.read_csv(fichier_historique)
                if fichier_historique.name.endswith(".csv")
                else pd.read_excel(fichier_historique)
            )
            df_propre = df_hist.iloc[:, :10].dropna()
            historiques = df_propre.astype(int).values.tolist()
            grilles_filtrees = filtrer_par_historique(
                grilles_brutes, historiques, seuil_exclu
            )
            grilles_finales = grilles_filtrees[:nb_grilles_demande]
        else:
            grilles_finales = grilles_brutes[:nb_grilles_demande]

        if grilles_finales:
            st.session_state.grilles_actives = grilles_finales
            st.success(
                f"{len(grilles_finales)} grilles conformes prêtes à l'exploitation."
            )

            # Formatage du tableau avec colonnes N1 à N10
            df_results = pd.DataFrame(
                grilles_finales, columns=[f"N{i+1}" for i in range(10)]
            )
            df_results.insert(
                0, "Grille", [f"G{i+1}" for i in range(len(grilles_finales))]
            )
            df_results["Somme"] = [sum(g) for g in grilles_finales]
            df_results["G1 [1-9]"] = [
                sum(1 for x in g if 1 <= x <= 9) for g in grilles_finales
            ]
            df_results["G2 [10-19]"] = [
                sum(1 for x in g if 10 <= x <= 19) for g in grilles_finales
            ]
            df_results["G3 [20-25]"] = [
                sum(1 for x in g if 20 <= x <= 25) for g in grilles_finales
            ]

            st.dataframe(df_results, use_container_width=True)

            st.download_button(
                label="📥 Exporter en fichier Excel (.xlsx)",
                data=convert_df_to_excel(
                    df_results, sheet_name="Grilles_Empiriques"
                ),
                file_name="grilles_empiriques.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_t1",
            )
        else:
            st.warning("Aucune grille trouvée avec ces contraintes.")

# ------------------------------------------------------------------------------
# ONGLET 2 : SYSTÈME RÉDUCTEUR MANDEL (AVEC EXPORT EXCEL)
# ------------------------------------------------------------------------------
with tab2:
    st.header("Couverture Combinatoire Optimale (Mandel)")

    col_r1, col_r2, col_r3 = st.columns(3)
    with col_r1:
        pool_reducteur_input = st.text_input(
            "Pool à couvrir :",
            "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25",
            key="pool_r_input",
        )
    with col_r2:
        garantie_cible = st.selectbox(
            "Garantie mathématique visée", [6, 7, 8], index=2
        )
    with col_r3:
        max_grilles_mandel = st.slider(
            "Nombre max de grilles à générer :",
            min_value=5,
            max_value=60,
            value=20,
        )
        forcer_decades_t2 = st.checkbox(
            "Forcer condition : ≥2 [1-9] et ≥2 [10-19]",
            value=True,
            key="decades_t2",
        )

    if st.button("Générer le Système Réducteur Mandel", type="primary"):
        pool_r = [int(x) for x in pool_reducteur_input.split() if x.isdigit()]
        pool_r = sorted(list(set(pool_r)))

        g1_count = len([x for x in pool_r if 1 <= x <= 9])
        g2_count = len([x for x in pool_r if 10 <= x <= 19])

        if len(pool_r) < 10:
            st.error("Le pool doit contenir au minimum 10 numéros.")
        elif forcer_decades_t2 and (g1_count < 2 or g2_count < 2):
            st.error(
                "Le pool doit contenir au moins 2 chiffres dans [1-9] et 2 dans [10-19]."
            )
        else:
            with st.spinner("Calcul de la condensation combinatoire..."):
                grilles_mandel = algorithme_glouton_mandel(
                    pool=pool_r,
                    taille_grille=10,
                    garantie=garantie_cible,
                    filtrer_decades=forcer_decades_t2,
                    max_grilles=max_grilles_mandel,
                )

            if grilles_mandel:
                st.session_state.grilles_actives = grilles_mandel
                st.success(
                    f"{len(grilles_mandel)} grilles réduites générées pour un objectif ≥ {garantie_cible}/10."
                )

                df_mandel = pd.DataFrame(
                    grilles_mandel, columns=[f"N{i+1}" for i in range(10)]
                )
                df_mandel.insert(
                    0, "Grille", [f"G{i+1}" for i in range(len(grilles_mandel))]
                )
                df_mandel["Somme"] = [sum(g) for g in grilles_mandel]
                df_mandel["G1 [1-9]"] = [
                    sum(1 for x in g if 1 <= x <= 9) for g in grilles_mandel
                ]
                df_mandel["G2 [10-19]"] = [
                    sum(1 for x in g if 10 <= x <= 19) for g in grilles_mandel
                ]
                df_mandel["G3 [20-25]"] = [
                    sum(1 for x in g if 20 <= x <= 25) for g in grilles_mandel
                ]

                st.dataframe(df_mandel, use_container_width=True)

                st.download_button(
                    label="📥 Exporter le Système Réducteur en Excel (.xlsx)",
                    data=convert_df_to_excel(
                        df_mandel, sheet_name="Mandel_Reducteur"
                    ),
                    file_name="systeme_reducteur_mandel.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_t2",
                )
            else:
                st.warning("Impossible de générer des grilles sous ces critères.")

# ==============================================================================
# MODULE DE VÉRIFICATION & EXTRACTION AUDIT
# ==============================================================================
st.divider()
st.subheader("Audit du Tirage & Extraction des Résultats")

if st.session_state.grilles_actives:
    nb_grilles = len(st.session_state.grilles_actives)
    st.info(f"**Jeu chargé en mémoire :** {nb_grilles} grilles prêtes pour l'audit.")

    col_in, col_calib, col_btn = st.columns([3, 1.5, 1])
    with col_in:
        tirage_test_input = st.text_input(
            "Saisis les 10 numéros du tirage (séparés par un espace) :",
            "2 5 11 14 18 20 21 22 24 25",
        )
    with col_calib:
        seuil_objectif = st.slider(
            "🎯 Objectif de bons numéros :",
            min_value=5,
            max_value=10,
            value=8,
            help="Définit le seuil pour identifier les grilles gagnantes.",
        )
    with col_btn:
        st.write("")
        st.write("")
        lancer_audit = st.button("Comparer au tirage", type="secondary")

    if lancer_audit:
        tirage_test = [int(x) for x in tirage_test_input.split() if x.isdigit()]

        if len(tirage_test) != 10:
            st.error("Erreur : saisis exactement 10 numéros.")
        elif len(set(tirage_test)) != 10:
            st.error("Erreur : aucun doublon n'est permis dans le tirage.")
        elif any(x < 1 or x > 25 for x in tirage_test):
            st.error("Erreur : les numéros doivent être compris entre 1 et 25.")
        else:
            set_gagnante = set(tirage_test)
            scores = [
                len(set(g).intersection(set_gagnante))
                for g in st.session_state.grilles_actives
            ]
            bilan = {k: scores.count(k) for k in range(0, 11)}

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("6 Bons", bilan.get(6, 0))
            m2.metric("7 Bons", bilan.get(7, 0))
            label_obj = (
                f"{seuil_objectif} Bons (Objectif)"
                if seuil_objectif not in [6, 7, 10]
                else f"{seuil_objectif} Bons"
            )
            m3.metric(label_obj, bilan.get(seuil_objectif, 0))
            m4.metric("9 Bons", bilan.get(9, 0))
            m5.metric("10/10", bilan.get(10, 0))

            total_succes = sum(bilan.get(k, 0) for k in range(seuil_objectif, 11))
            if total_succes > 0:
                pct = (total_succes / nb_grilles) * 100
                st.success(
                    f"🎯 Objectif atteint : {total_succes} grille(s) sur {nb_grilles} ({pct:.1f}%) ont au moins {seuil_objectif} bons numéros !"
                )
            else:
                st.warning(f"Aucune grille n'atteint {seuil_objectif}/10 sur ce tirage.")

            # Construction du tableau d'audit exportable
            audit_rows = []
            for idx, g in enumerate(st.session_state.grilles_actives, 1):
                communs = sorted(list(set(g).intersection(set_gagnante)))
                score_grille = len(communs)
                statut = (
                    f"GAGNANT (≥{seuil_objectif})"
                    if score_grille >= seuil_objectif
                    else ("PRIMÉ" if score_grille >= 6 else "-")
                )

                row_dict = {"Grille": f"G{idx}"}
                for num_i, val in enumerate(g, 1):
                    row_dict[f"N{num_i}"] = val
                row_dict["Score"] = score_grille
                row_dict["Numéros Trouvés"] = (
                    ", ".join(str(x) for x in communs) if communs else ""
                )
                row_dict["Statut"] = statut
                audit_rows.append(row_dict)

            df_audit = pd.DataFrame(audit_rows)
            st.dataframe(df_audit, use_container_width=True)

            st.download_button(
                label="📥 Télécharger l'Audit complet en Excel (.xlsx)",
                data=convert_df_to_excel(df_audit, sheet_name="Audit_Tirage"),
                file_name="audit_tirage_resultats.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="dl_audit",
            )
else:
    st.caption("Génère d'abord des grilles dans l'un des deux onglets ci-dessus pour activer l'audit.")
