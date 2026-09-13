import io
import random
from typing import List, Set
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
# FONCTIONS LOGIQUES ET ALGORITHMES (GÉNÉRATION SÉLECTIVE)
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


def generer_grilles_mandel_filtrees(
    pool: List[int],
    base_initiale: List[int],
    nb_grilles: int = 15,
    forcer_base: bool = True,
    min_sum: int = 80,
    max_sum: int = 180,
    cible_10: int = -1,
    cible_20: int = -1,
    filtrer_mandel: bool = True,
) -> List[List[int]]:
    """Génération sélective et rapide par échantillonnage ciblé sous contraintes :

    - Au moins 2 chiffres dans [1-9] (si Mandel actif)
    - Au moins 2 chiffres dans [10-19] (si Mandel actif)
    - Filtre de somme, ancrage sur base et ciblage décadaire précis
    """
    g1 = [x for x in pool if 1 <= x <= 9]
    g2 = [x for x in pool if 10 <= x <= 19]
    base_set = set(base_initiale) if base_initiale else set()

    if filtrer_mandel and (len(g1) < 2 or len(g2) < 2):
        return []

    grilles = []
    seen = set()
    attempts = 0
    max_attempts = 30000

    while len(grilles) < nb_grilles and attempts < max_attempts:
        attempts += 1
        ticket = set()

        # 1. Respect des contraintes décadaires minimales Mandel
        if filtrer_mandel:
            k1 = random.choice([2, 3]) if len(g1) >= 3 else 2
            k2 = random.choice([2, 3]) if len(g2) >= 3 else 2
            ticket.update(random.sample(g1, k1))
            ticket.update(random.sample(g2, k2))

        # 2. Compléter jusqu'à 10 chiffres avec le pool restant
        rest_pool = [x for x in pool if x not in ticket]
        needed = 10 - len(ticket)
        if len(rest_pool) < needed:
            continue
        ticket.update(random.sample(rest_pool, needed))

        combi = sorted(list(ticket))

        # 3. Filtre sur la somme
        somme_combi = sum(combi)
        if somme_combi < min_sum or (max_sum > 0 and somme_combi > max_sum):
            continue

        # 4. Filtre d'ancrage sur la base
        if forcer_base and len(set(combi).intersection(base_set)) < 3:
            continue

        # 5. Filtres topologiques exacts
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


def convert_df_to_excel(df):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False, sheet_name="Grilles")
    return output.getvalue()


# ==============================================================================
# INTERFACE UTILISATEUR
# ==============================================================================

st.title("Système de Génération & Réduction Mathématique")
tab1, tab2 = st.tabs(
    ["Moteur 1 : Filtrage Empirique", "Moteur 2 : Système Réducteur Mandel"]
)

# ------------------------------------------------------------------------------
# ONGLET 1 : FILTRAGE EMPIRIQUE (GÉNÉRATION CIBLÉE)
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
        nb_grilles_demande = st.number_input(
            "Nombre de grilles à générer", value=15, min_value=1, max_value=200
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
        "Importer l'historique des tirages", type=["csv", "xlsx"]
    )

    if st.button("Lancer le Moteur Sélectif", type="primary"):
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

        with st.spinner("Génération sélective des grilles conformes..."):
            # On génère un surplus raisonnable pour absorber l'éventuel filtre d'historique
            quota_recherche = (
                nb_grilles_demande * 4
                if fichier_historique
                else nb_grilles_demande
            )
            grilles_brutes = generer_grilles_mandel_filtrees(
                pool=pool_final,
                base_initiale=base_liste,
                nb_grilles=quota_recherche,
                forcer_base=forcer_base,
                min_sum=min_sum,
                max_sum=max_sum,
                cible_10=cible_10,
                cible_20=cible_20,
                filtrer_mandel=filtre_mandel_tab1,
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
                f"{len(grilles_finales)} grilles conformes générées instantanément."
            )

            df_results = pd.DataFrame(
                grilles_finales, columns=[f"N{i+1}" for i in range(10)]
            )
            df_results["G1 [1-9]"] = [
                sum(1 for x in g if 1 <= x <= 9) for g in grilles_finales
            ]
            df_results["G2 [10-19]"] = [
                sum(1 for x in g if 10 <= x <= 19) for g in grilles_finales
            ]
            df_results["Somme"] = [sum(g) for g in grilles_finales]

            st.dataframe(df_results, use_container_width=True)
            st.download_button(
                "Télécharger les grilles (Excel)",
                convert_df_to_excel(df_results),
                "grilles_selectives.xlsx",
            )
        else:
            st.warning(
                "Aucune grille trouvée. Vérifie que le pool contient au moins 2 chiffres dans [1-9] et 2 dans [10-19] ou relâche légèrement les contraintes de somme/base."
            )

# ------------------------------------------------------------------------------
# ONGLET 2 : SYSTÈME RÉDUCTEUR MANDEL DIRECT
# ------------------------------------------------------------------------------
with tab2:
    st.header("Couverture Combinatoire Optimale (Mandel)")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        pool_reducteur_input = st.text_input(
            "Pool de 25 numéros :", " ".join(str(x) for x in range(1, 26))
        )
    with col_r2:
        nb_grilles_mandel = st.slider(
            "Nombre de grilles à générer :",
            min_value=5,
            max_value=60,
            value=15,
        )

    if st.button("Générer le Jeu Réducteur Mandel", type="primary"):
        pool_r = [int(x) for x in pool_reducteur_input.split() if x.isdigit()]
        pool_r = sorted(list(set(pool_r)))

        if len(pool_r) < 10:
            st.error("Le pool doit contenir au minimum 10 numéros.")
        else:
            grilles_mandel = generer_grilles_mandel_filtrees(
                pool=pool_r,
                base_initiale=[],
                nb_grilles=nb_grilles_mandel,
                forcer_base=False,
                min_sum=0,
                max_sum=0,
                filtrer_mandel=True,
            )

            st.session_state.grilles_actives = grilles_mandel
            st.success(
                f"{len(grilles_mandel)} grilles générées avec respect strict des plages décadaires."
            )

            df_mandel = pd.DataFrame(
                grilles_mandel, columns=[f"N{i+1}" for i in range(10)]
            )
            df_mandel["G1 [1-9]"] = [
                sum(1 for x in g if 1 <= x <= 9) for g in grilles_mandel
            ]
            df_mandel["G2 [10-19]"] = [
                sum(1 for x in g if 10 <= x <= 19) for g in grilles_mandel
            ]
            st.dataframe(df_mandel, use_container_width=True)
            st.download_button(
                "Télécharger (Excel)",
                convert_df_to_excel(df_mandel),
                "systeme_mandel.xlsx",
            )

# ==============================================================================
# MODULE D'AUDIT : COMPARAISON AVEC LA BONNE COMBINAISON
# ==============================================================================
st.divider()
st.subheader("Audit & Vérification des Grilles en Mémoire")

if st.session_state.grilles_actives:
    nb_grilles = len(st.session_state.grilles_actives)
    st.info(f"**Jeu chargé :** {nb_grilles} grilles prêtes pour l'audit.")

    col_in, col_btn = st.columns([3, 1])
    with col_in:
        tirage_test_input = st.text_input(
            "Saisis la combinaison gagnante (10 numéros) :",
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
            st.error("Erreur : saisis exactement 10 numéros distincts.")
        elif len(set(tirage_test)) != 10:
            st.error("Erreur : la combinaison ne doit comporter aucun doublon.")
        elif any(x < 1 or x > 25 for x in tirage_test):
            st.error(
                "Périmètre invalide : tous les numéros doivent être compris entre 1 et 25."
            )
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
                    "Aucune grille n'atteint 8/10 ou plus sur ce tirage."
                )

            # Tableau visuel avec mise en valeur
            table_rows = []
            for idx, g in enumerate(st.session_state.grilles_actives, 1):
                communs = sorted(list(set(g).intersection(set_gagnante)))
                c1 = sum(1 for x in g if 1 <= x <= 9)
                c2 = sum(1 for x in g if 10 <= x <= 19)

                # Formatage visuel : les numéros gagnants sont encadrés par des étoiles
                visuel = "  ".join(
                    f"*{x:02d}*" if x in set_gagnante else f"{x:02d}" for x in g
                )

                table_rows.append(
                    {
                        "Grille": f"#{idx:02d}",
                        "Composition (étoiles = gagnants)": visuel,
                        "G1 [1-9]": c1,
                        "G2 [10-19]": c2,
                        "Numéros trouvés": " - ".join(f"{x:02d}" for x in communs)
                        if communs
                        else "-",
                        "Score": f"{len(communs)} / 10",
                        "Statut": "🌟 GAGNANT (≥8/10)"
                        if len(communs) >= 8
                        else ("PRIMÉ (6-7)" if len(communs) >= 6 else "-"),
                    }
                )

            df_detail = pd.DataFrame(table_rows)
            st.dataframe(df_detail, use_container_width=True)
else:
    st.caption(
        "Génère d'abord des grilles dans l'un des deux onglets ci-dessus pour activer ce module."
    )
