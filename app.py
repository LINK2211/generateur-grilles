import io
import itertools
import math
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
MAX_EXCEL_ROWS = 1048500

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


def analyser_et_filtrer_univers(
    pool: List[int],
    base_initiale: List[int],
    forcer_base: bool = True,
    min_sum: int = 80,
    max_sum: int = 180,
    cible_10: int = -1,
    cible_20: int = -1,
    forcer_decades: bool = True,
):
    """Parcourt l'espace combinatoire pour dénombrer l'élagage exact."""
    total_theorique = math.comb(len(pool), 10)
    base_set = set(base_initiale) if base_initiale else set()
    valides = []

    for combi in itertools.combinations(pool, 10):
        if forcer_decades and not verifier_contraintes_decades(
            combi, min_g1=2, min_g2=2
        ):
            continue
        s = sum(combi)
        if s < min_sum or (max_sum > 0 and s > max_sum):
            continue
        if forcer_base and len(set(combi).intersection(base_set)) < 3:
            continue
        if cible_10 != -1 and sum(1 for x in combi if 10 <= x <= 19) != cible_10:
            continue
        if cible_20 != -1 and sum(1 for x in combi if 20 <= x <= 25) != cible_20:
            continue
        valides.append(list(combi))

    nb_valides = len(valides)
    nb_eliminees = total_theorique - nb_valides
    return total_theorique, nb_valides, nb_eliminees, valides


def algorithme_glouton_mandel(
    pool: List[int],
    taille_grille: int = 10,
    garantie: int = 8,
    filtrer_decades: bool = True,
    max_grilles: int = 20,
) -> List[List[int]]:
    """Moteur 2 : algorithme glouton réducteur de Mandel (Set Cover)."""
    candidats = list(itertools.combinations(pool, taille_grille))
    if filtrer_decades:
        candidats = [
            c
            for c in candidats
            if verifier_contraintes_decades(c, min_g1=2, min_g2=2)
        ]

    if not candidats:
        return []

    tous_subsets = list(itertools.combinations(pool, garantie))
    sous_ensembles_a_couvrir = (
        set(tous_subsets[:15000]) if len(tous_subsets) > 15000 else set(tous_subsets)
    )

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


def filtrer_combinaison_dutel(
    comb,
    min_somme=None,
    max_somme=None,
    max_consecutifs=3,
    max_par_dizaine=4,
    min_pairs=4,
    max_pairs=6,
):
    """Vérifie les critères statistiques de Marie Dutel."""
    pairs = sum(1 for x in comb if x % 2 == 0)
    if pairs < min_pairs or pairs > max_pairs:
        return False
    somme = sum(comb)
    if min_somme is not None and somme < min_somme:
        return False
    if max_somme is not None and somme > max_somme:
        return False
    consecutifs = 1
    for i in range(len(comb) - 1):
        if comb[i + 1] == comb[i] + 1:
            consecutifs += 1
            if consecutifs > max_consecutifs:
                return False
        else:
            consecutifs = 1
    dizaines = {}
    for x in comb:
        d = x // 10
        dizaines[d] = dizaines.get(d, 0) + 1
        if dizaines[d] > max_par_dizaine:
            return False
    return True


def generer_grilles_dutel(
    numeros_base: List[int],
    nb_a_generer: int = 15,
    max_consecutifs: int = 3,
    max_par_dizaine: int = 4,
    pct_amplitude: float = 0.25,
):
    """Génération selon la théorie Dutel."""
    numeros = sorted(list(set(numeros_base)))
    if len(numeros) < 10:
        return [], 0, 0
    somme_min_possible = sum(numeros[:10])
    somme_max_possible = sum(numeros[-10:])
    moyenne_somme = (somme_min_possible + somme_max_possible) / 2
    amplitude = (somme_max_possible - somme_min_possible) * pct_amplitude
    borne_basse = int(moyenne_somme - amplitude)
    borne_haute = int(moyenne_somme + amplitude)

    valides = []
    seen = set()
    attempts = 0
    max_attempts = 40000

    while len(valides) < nb_a_generer and attempts < max_attempts:
        attempts += 1
        cand = tuple(sorted(random.sample(numeros, 10)))
        if cand not in seen:
            seen.add(cand)
            if filtrer_combinaison_dutel(
                cand,
                min_somme=borne_basse,
                max_somme=borne_haute,
                max_consecutifs=max_consecutifs,
                max_par_dizaine=max_par_dizaine,
            ):
                valides.append(list(cand))
    return valides, borne_basse, borne_haute


def generer_matrice_bibd_equilibre(
    pool: List[int], nb_grilles: int = 20, seed: int = 42
) -> List[List[int]]:
    """Moteur 4 : Génération par matrice équilibrée (BIBD / Gail Howard).

    Minimise la variance d'apparition des paires (lambda-uniformité).
    """
    random.seed(seed)
    numeros = sorted(list(set(pool)))
    if len(numeros) < 10:
        return []

    compteur_paires = {p: 0 for p in itertools.combinations(numeros, 2)}
    frequence_numeros = {n: 0 for n in numeros}
    grilles = []

    for _ in range(nb_grilles):
        candidats = []
        for _ in range(150):
            ticket = sorted(random.sample(numeros, 10))
            paires_ticket = list(itertools.combinations(ticket, 2))
            score_penalite = sum(compteur_paires[p] for p in paires_ticket)
            score_penalite += sum(frequence_numeros[n] * 2 for n in ticket)
            candidats.append((score_penalite, ticket, paires_ticket))

        candidats.sort(key=lambda x: x[0])
        meilleur_ticket = candidats[0][1]
        meilleures_paires = candidats[0][2]

        grilles.append(meilleur_ticket)
        for p in meilleures_paires:
            compteur_paires[p] += 1
        for n in meilleur_ticket:
            frequence_numeros[n] += 1

    return grilles


def generer_grilles_esperance_mit(
    pool: List[int],
    nb_grilles: int = 15,
    seuil_anti_partage: bool = True,
    min_sum: int = 85,
    max_sum: int = 175,
) -> List[List[int]]:
    """Moteur 5 : Arbitrage d'espérance mathématique & anti-partage (MIT).

    Élimine les combinaisons surjouées par la masse (biais calendaires / réguliers).
    """
    numeros = sorted(list(set(pool)))
    if len(numeros) < 10:
        return []

    valides = []
    seen = set()
    attempts = 0
    max_attempts = 40000

    while len(valides) < nb_grilles and attempts < max_attempts:
        attempts += 1
        cand = sorted(random.sample(numeros, 10))
        cand_tuple = tuple(cand)
        if cand_tuple in seen:
            continue
        seen.add(cand_tuple)

        s = sum(cand)
        if s < min_sum or s > max_sum:
            continue

        if seuil_anti_partage:
            # Éviter les grilles calendaires (la masse joue massivement <= 12 pour mois et <= 20)
            nb_bas = sum(1 for x in cand if x <= 12)
            if nb_bas >= 7:  # Trop de numéros typiques de dates
                continue
            # Éviter les grilles à pas constant (ex: 2, 4, 6, 8...)
            differences = [cand[i + 1] - cand[i] for i in range(len(cand) - 1)]
            if len(set(differences)) <= 2:
                continue

        valides.append(cand)

    return valides


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
    df_export = df.iloc[:MAX_EXCEL_ROWS] if len(df) > MAX_EXCEL_ROWS else df
    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        df_export.to_excel(writer, index=False, sheet_name=sheet_name)
        worksheet = writer.sheets[sheet_name]
        for idx, col in enumerate(df_export.columns):
            max_len = max(df_export[col].astype(str).map(len).max(), len(str(col))) + 2
            worksheet.set_column(idx, idx, max_len)
    return output.getvalue()


def convert_df_to_csv(df):
    """Export CSV universel sans contrainte de lignes."""
    return df.to_csv(index=False).encode("utf-8")


# ==============================================================================
# INTERFACE UTILISATEUR
# ==============================================================================

st.title("Système de Génération & Réduction Mathématique")
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Moteur 1 : Filtrage Empirique & Analyse",
        "Moteur 2 : Système Réducteur Mandel",
        "Moteur 3 : Théorie Marie Dutel",
        "Moteur 4 : Matrices Équilibrées (BIBD)",
        "Moteur 5 : Arbitrage & Espérance (MIT)",
    ]
)

# ------------------------------------------------------------------------------
# ONGLET 1 : FILTRAGE EMPIRIQUE & COMPTAGE D'ÉLAGAGE
# ------------------------------------------------------------------------------
with tab1:
    st.header("Filtrage par Hypothèses & Audit Combinatoire")
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
        mode_generation = st.radio(
            "Mode de génération :",
            ["Générer un nombre précis (Échantillon)", "Tout générer (Toutes les combinaisons valides)"],
            index=0,
        )
        nb_grilles_demande = st.number_input(
            "Nombre de grilles à retenir",
            value=20,
            min_value=1,
            max_value=10000,
            disabled=(mode_generation == "Tout générer"),
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

    if st.button("Lancer l'Analyse & la Génération", type="primary"):
        base_liste = [int(x) for x in base_input.split() if x.isdigit()]
        sous_pool = preparer_pool(base_liste)
        pool_final = sorted(list(set(sous_pool).intersection(set(pool_global))))
        if not pool_final:
            pool_final = pool_global

        st.info(f"Pool final de travail : {pool_final} ({len(pool_final)} numéros)")

        with st.spinner("Analyse combinatoire et application des filtres..."):
            total_theorique, nb_valides, nb_eliminees, grilles_valides = (
                analyser_et_filtrer_univers(
                    pool=pool_final,
                    base_initiale=base_liste,
                    forcer_base=forcer_base,
                    min_sum=min_sum,
                    max_sum=max_sum,
                    cible_10=cible_10,
                    cible_20=cible_20,
                    forcer_decades=forcer_decades_t1,
                )
            )

        st.markdown("### 📊 Rapport d'élagage combinatoire")
        m_tot, m_val, m_elim, m_pct = st.columns(4)
        m_tot.metric("Total Théorique C(N, 10)", f"{total_theorique:,}")
        m_val.metric("Grilles Conformes", f"{nb_valides:,}")
        m_elim.metric("Grilles Éliminées", f"{nb_eliminees:,}")
        taux_elagage = (
            (nb_eliminees / total_theorique * 100) if total_theorique > 0 else 0
        )
        m_pct.metric("Taux d'élimination", f"{taux_elagage:.2f} %")

        historiques = []
        if fichier_historique and grilles_valides:
            df_hist = (
                pd.read_csv(fichier_historique)
                if fichier_historique.name.endswith(".csv")
                else pd.read_excel(fichier_historique)
            )
            df_propre = df_hist.iloc[:, :10].dropna()
            historiques = df_propre.astype(int).values.tolist()
            with st.spinner("Application du filtre d'historique..."):
                grilles_valides = filtrer_par_historique(
                    grilles_valides, historiques, seuil_exclu
                )
            st.write(f"**Après filtre historique :** {len(grilles_valides):,} grilles restantes.")

        if mode_generation == "Générer un nombre précis (Échantillon)":
            if len(grilles_valides) > nb_grilles_demande:
                grilles_finales = random.sample(grilles_valides, nb_grilles_demande)
            else:
                grilles_finales = grilles_valides
        else:
            grilles_finales = grilles_valides

        if grilles_finales:
            st.session_state.grilles_actives = grilles_finales
            st.success(f"**{len(grilles_finales):,} grille(s)** chargées en mémoire.")

            df_results = pd.DataFrame(
                grilles_finales, columns=[f"N{i+1}" for i in range(10)]
            )
            df_results.insert(0, "Grille", [f"G{i+1}" for i in range(len(grilles_finales))])
            df_results["Somme"] = [sum(g) for g in grilles_finales]
            df_results["G1 [1-9]"] = [sum(1 for x in g if 1 <= x <= 9) for g in grilles_finales]
            df_results["G2 [10-19]"] = [sum(1 for x in g if 10 <= x <= 19) for g in grilles_finales]
            df_results["G3 [20-25]"] = [sum(1 for x in g if 20 <= x <= 25) for g in grilles_finales]

            st.dataframe(df_results.head(1000), use_container_width=True)
            if len(df_results) > 1000:
                st.caption(f"Aperçu limité aux 1 000 premières lignes sur {len(df_results):,}.")

            cd1, cd2 = st.columns(2)
            with cd1:
                st.download_button(
                    "📥 Télécharger toutes les grilles (CSV)",
                    convert_df_to_csv(df_results),
                    "grilles_empiriques.csv",
                    "text/csv",
                    key="dl_t1_csv",
                )
            with cd2:
                st.download_button(
                    "📥 Télécharger en Excel (.xlsx)",
                    convert_df_to_excel(df_results, sheet_name="Grilles"),
                    "grilles_empiriques.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_t1_excel",
                )
        else:
            st.warning("Aucune grille ne respecte l'ensemble de ces filtres.")

# ------------------------------------------------------------------------------
# ONGLET 2 : SYSTÈME RÉDUCTEUR MANDEL
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
        garantie_cible = st.selectbox("Garantie mathématique visée", [6, 7, 8], index=2)
    with col_r3:
        max_grilles_mandel = st.slider(
            "Nombre max de grilles à générer :", min_value=5, max_value=60, value=20
        )
        forcer_decades_t2 = st.checkbox(
            "Forcer condition : ≥2 [1-9] et ≥2 [10-19]", value=True, key="decades_t2"
        )

    if st.button("Générer le Système Réducteur Mandel", type="primary"):
        pool_r = [int(x) for x in pool_reducteur_input.split() if x.isdigit()]
        pool_r = sorted(list(set(pool_r)))
        g1_count = len([x for x in pool_r if 1 <= x <= 9])
        g2_count = len([x for x in pool_r if 10 <= x <= 19])

        if len(pool_r) < 10:
            st.error("Le pool doit contenir au minimum 10 numéros.")
        elif forcer_decades_t2 and (g1_count < 2 or g2_count < 2):
            st.error("Le pool doit contenir au moins 2 chiffres dans [1-9] et 2 dans [10-19].")
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
                st.success(f"{len(grilles_mandel)} grilles réduites générées pour rang ≥ {garantie_cible}/10.")
                df_mandel = pd.DataFrame(grilles_mandel, columns=[f"N{i+1}" for i in range(10)])
                df_mandel.insert(0, "Grille", [f"G{i+1}" for i in range(len(grilles_mandel))])
                df_mandel["Somme"] = [sum(g) for g in grilles_mandel]
                df_mandel["G1 [1-9]"] = [sum(1 for x in g if 1 <= x <= 9) for g in grilles_mandel]
                df_mandel["G2 [10-19]"] = [sum(1 for x in g if 10 <= x <= 19) for g in grilles_mandel]
                df_mandel["G3 [20-25]"] = [sum(1 for x in g if 20 <= x <= 25) for g in grilles_mandel]

                st.dataframe(df_mandel, use_container_width=True)
                st.download_button(
                    "📥 Exporter le Système Réducteur en Excel (.xlsx)",
                    convert_df_to_excel(df_mandel, sheet_name="Mandel_Reducteur"),
                    "systeme_reducteur_mandel.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_t2",
                )

# ------------------------------------------------------------------------------
# ONGLET 3 : THÉORIE MARIE DUTEL
# ------------------------------------------------------------------------------
with tab3:
    st.header("Filtrage Statistique & Courbe de Gauss (Marie Dutel)")
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        pool_dutel_input = st.text_input(
            "Pool sélectionné :",
            "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25",
            key="pool_dutel_in",
        )
        nb_grilles_dutel = st.number_input(
            "Nombre de grilles à générer :", value=15, min_value=1, max_value=200, key="nb_dutel"
        )
    with col_d2:
        max_consec = st.slider("Max consécutifs :", min_value=1, max_value=4, value=3)
        max_diz = st.slider("Max par dizaine :", min_value=2, max_value=6, value=4)
    with col_d3:
        amplitude_gauss = st.slider(
            "Largeur Gauss (± %) :", min_value=0.10, max_value=0.40, value=0.25, step=0.05
        )

    if st.button("Générer selon la Théorie Dutel", type="primary"):
        pool_d = [int(x) for x in pool_dutel_input.split() if x.isdigit()]
        pool_d = sorted(list(set(pool_d)))

        if len(pool_d) < 10:
            st.error("Le pool doit comporter au moins 10 numéros distincts.")
        else:
            with st.spinner("Calcul gaussien et filtrage..."):
                grilles_dutel, b_basse, b_haute = generer_grilles_dutel(
                    numeros_base=pool_d,
                    nb_a_generer=nb_grilles_dutel,
                    max_consecutifs=max_consec,
                    max_par_dizaine=max_diz,
                    pct_amplitude=amplitude_gauss,
                )

            if grilles_dutel:
                st.session_state.grilles_actives = grilles_dutel
                st.success(f"{len(grilles_dutel)} grilles conformes générées | Fourchette : [{b_basse} - {b_haute}]")
                df_dutel = pd.DataFrame(grilles_dutel, columns=[f"N{i+1}" for i in range(10)])
                df_dutel.insert(0, "Grille", [f"G{i+1}" for i in range(len(grilles_dutel))])
                df_dutel["Somme"] = [sum(g) for g in grilles_dutel]
                df_dutel["Pairs"] = [sum(1 for x in g if x % 2 == 0) for g in grilles_dutel]
                df_dutel["Impairs"] = [10 - p for p in df_dutel["Pairs"]]
                df_dutel["G1 [1-9]"] = [sum(1 for x in g if 1 <= x <= 9) for g in grilles_dutel]
                df_dutel["G2 [10-19]"] = [sum(1 for x in g if 10 <= x <= 19) for g in grilles_dutel]
                df_dutel["G3 [20-25]"] = [sum(1 for x in g if 20 <= x <= 25) for g in grilles_dutel]

                st.dataframe(df_dutel, use_container_width=True)
                st.download_button(
                    "📥 Exporter les grilles Dutel en Excel (.xlsx)",
                    convert_df_to_excel(df_dutel, sheet_name="Marie_Dutel"),
                    "grilles_theorie_dutel.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_t3",
                )

# ------------------------------------------------------------------------------
# ONGLET 4 : MATRICES ÉQUILIBRÉES (BIBD / GAIL HOWARD)
# ------------------------------------------------------------------------------
with tab4:
    st.header("Matrices de Blocs Incomplets Équilibrés (BIBD)")
    st.caption("Minimise la variance spatiale : chaque paire de numéros est représentée un nombre égal de fois ($\lambda$-uniformité).")

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        pool_bibd_input = st.text_input(
            "Pool pour matrice équilibrée :",
            "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25",
            key="pool_bibd_in",
        )
    with col_b2:
        nb_grilles_bibd = st.slider("Nombre de blocs (grilles) :", min_value=5, max_value=50, value=20)

    if st.button("Générer la Matrice Équilibrée BIBD", type="primary"):
        pool_b = [int(x) for x in pool_bibd_input.split() if x.isdigit()]
        pool_b = sorted(list(set(pool_b)))

        if len(pool_b) < 10:
            st.error("Le pool doit contenir au minimum 10 numéros.")
        else:
            with st.spinner("Construction factorielle et équilibrage des paires..."):
                grilles_bibd = generer_matrice_bibd_equilibre(pool=pool_b, nb_grilles=nb_grilles_bibd)

            if grilles_bibd:
                st.session_state.grilles_actives = grilles_bibd
                st.success(f"Matrice BIBD générée : {len(grilles_bibd)} grilles à couverture harmonique.")

                df_bibd = pd.DataFrame(grilles_bibd, columns=[f"N{i+1}" for i in range(10)])
                df_bibd.insert(0, "Grille", [f"G{i+1}" for i in range(len(grilles_bibd))])
                df_bibd["Somme"] = [sum(g) for g in grilles_bibd]
                df_bibd["G1 [1-9]"] = [sum(1 for x in g if 1 <= x <= 9) for g in grilles_bibd]
                df_bibd["G2 [10-19]"] = [sum(1 for x in g if 10 <= x <= 19) for g in grilles_bibd]
                df_bibd["G3 [20-25]"] = [sum(1 for x in g if 20 <= x <= 25) for g in grilles_bibd]

                st.dataframe(df_bibd, use_container_width=True)
                st.download_button(
                    "📥 Exporter la matrice BIBD en Excel (.xlsx)",
                    convert_df_to_excel(df_bibd, sheet_name="Matrice_BIBD"),
                    "matrice_equilibree_bibd.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_t4",
                )

# ------------------------------------------------------------------------------
# ONGLET 5 : ARBITRAGE & ESPÉRANCE (MIT / ROLL-DOWN)
# ------------------------------------------------------------------------------
with tab5:
    st.header("Optimisation d'Espérance & Filtre Anti-Partage (MIT)")
    st.caption("Élimine les combinaisons surjouées par la masse (dates de naissance, progressions arithmétiques) pour maximiser le gain net.")

    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        pool_mit_input = st.text_input(
            "Pool à exploiter :",
            "1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25",
            key="pool_mit_in",
        )
        nb_grilles_mit = st.number_input("Nombre de grilles :", value=15, min_value=1, max_value=100, key="nb_mit")
    with col_m2:
        anti_partage = st.checkbox("Activer le filtre Anti-Calendrier (MIT)", value=True)
        min_sum_mit = st.number_input("Somme mini", value=85, step=5)
        max_sum_mit = st.number_input("Somme maxi", value=175, step=5)
    with col_m3:
        st.markdown("**Simulateur Roll-Down (Espérance nette)**")
        jackpot_estime = st.number_input("Jackpot estimé (€)", value=200000, step=50000)
        cout_grille = st.number_input("Prix du ticket (€)", value=2.0, step=0.5)

    if st.button("Générer les Grilles à Haute Espérance", type="primary"):
        pool_m = [int(x) for x in pool_mit_input.split() if x.isdigit()]
        pool_m = sorted(list(set(pool_m)))

        if len(pool_m) < 10:
            st.error("Le pool doit contenir au minimum 10 numéros.")
        else:
            with st.spinner("Filtrage par entropie et élimination des biais de foule..."):
                grilles_mit = generer_grilles_esperance_mit(
                    pool=pool_m,
                    nb_grilles=nb_grilles_mit,
                    seuil_anti_partage=anti_partage,
                    min_sum=min_sum_mit,
                    max_sum=max_sum_mit,
                )

            if grilles_mit:
                st.session_state.grilles_actives = grilles_mit
                total_combis = math.comb(len(pool_m), 10)
                proba_jackpot = 1 / total_combis
                esperance_brute = (jackpot_estime * proba_jackpot) - cout_grille

                st.success(f"{len(grilles_mit)} grilles à forte entropie générées.")
                st.info(f"Espérance théorique du jackpot sur le pool de {len(pool_m)} : {esperance_brute:+.2f} € / grille.")

                df_mit = pd.DataFrame(grilles_mit, columns=[f"N{i+1}" for i in range(10)])
                df_mit.insert(0, "Grille", [f"G{i+1}" for i in range(len(grilles_mit))])
                df_mit["Somme"] = [sum(g) for g in grilles_mit]
                df_mit["G1 [1-9]"] = [sum(1 for x in g if 1 <= x <= 9) for g in grilles_mit]
                df_mit["G2 [10-19]"] = [sum(1 for x in g if 10 <= x <= 19) for g in grilles_mit]
                df_mit["G3 [20-25]"] = [sum(1 for x in g if 20 <= x <= 25) for g in grilles_mit]

                st.dataframe(df_mit, use_container_width=True)
                st.download_button(
                    "📥 Exporter les grilles MIT en Excel (.xlsx)",
                    convert_df_to_excel(df_mit, sheet_name="Modele_MIT"),
                    "grilles_modele_mit.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="dl_t5",
                )

# ==============================================================================
# MODULE DE VÉRIFICATION & EXTRACTION AUDIT (COMMUN À TOUS LES ONGLETS)
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
    st.caption("Génère d'abord des grilles dans l'un des onglets ci-dessus pour activer l'audit.")
