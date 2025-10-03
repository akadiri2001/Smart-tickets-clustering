import pandas as pd
import numpy as np
import re
import string
import unicodedata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Nettoyage de texte : minuscule, suppression des accents et de la ponctuation
def nettoyer_texte(texte):
    texte = texte.lower()
    texte = unicodedata.normalize('NFD', texte).encode('ascii', 'ignore').decode("utf-8")
    texte = texte.translate(str.maketrans('', '', string.punctuation))
    return texte

# Chargement des données depuis un fichier Excel
df = pd.read_excel("Export de Demandes.xlsx")

# Nettoyage des colonnes texte
df["Titre"] = df["Titre"].astype(str).apply(nettoyer_texte)
df["Description"] = df["Description"].astype(str).apply(nettoyer_texte)

# Vectorisation TF-IDF des titres et descriptions
vectorizer_titre = TfidfVectorizer(stop_words='french', ngram_range=(1,2))
vectorizer_desc = TfidfVectorizer(stop_words='french', ngram_range=(1,2))

X_titre = vectorizer_titre.fit_transform(df["Titre"])
X_desc = vectorizer_desc.fit_transform(df["Description"])

# Calcul des similarités cosinus entre tous les tickets
sim_titre = cosine_similarity(X_titre)
sim_desc = cosine_similarity(X_desc)

# Fonction pour calculer la cohérence moyenne d’un cluster
def coherence_par_cluster(clusters, sim_matrix):
    scores = []
    for label, cluster in enumerate(clusters):
        if len(cluster) > 1:
            pairs = [(i,j) for i in cluster for j in cluster if i < j]
            similarities = [sim_matrix[i][j] for i,j in pairs]
            moyenne = np.mean(similarities) if similarities else 0
        else:
            moyenne = 0
        scores.append({
            "cluster": label,
            "taille": len(cluster),
            "coherence_moyenne": round(moyenne, 4)
        })
    return scores

# Recherche du meilleur seuil de regroupement
print("Recherche du meilleur seuil...")

best_seuil = None
best_coherence = -np.inf

# Fonction de clustering basée sur un seuil de similarité
def clustering_par_seuil(seuil):
    clusters_tmp = []

    for i in range(len(sim_titre)):
        meilleur_cluster = None
        meilleure_similarite = 0

        for cluster in clusters_tmp:
            similarites = [ (sim_titre[i][j] + sim_desc[i][j]) / 2 for j in cluster ]
            moyenne_sim = sum(similarites) / len(similarites)

            if moyenne_sim > meilleure_similarite and moyenne_sim >= seuil:
                meilleure_similarite = moyenne_sim
                meilleur_cluster = cluster

        if meilleur_cluster is not None:
            meilleur_cluster.append(i)
        else:
            clusters_tmp.append([i])

    return clusters_tmp

# Calcul de la cohérence pondérée (pondérée par la taille des clusters)
def coherence_ponderee(seuil):
    clusters_tmp = clustering_par_seuil(seuil)
    scores_tmp = coherence_par_cluster(clusters_tmp, (sim_titre + sim_desc) / 2)
    total = sum(s['taille'] for s in scores_tmp)
    if total == 0:
        return 0, clusters_tmp
    return sum(s['coherence_moyenne'] * s['taille'] for s in scores_tmp) / total, clusters_tmp

# Boucle de test sur plusieurs seuils pour trouver le meilleur
for seuil_test in np.arange(0.05, 0.1501, 0.005):
    coh, clust = coherence_ponderee(seuil_test)
    if coh > best_coherence:
        best_coherence = coh
        best_seuil = seuil_test
        best_clusters = clust

print(f"Meilleur seuil : {best_seuil:.4f} avec cohérence moyenne pondérée : {best_coherence:.4f}")

# Attribution des clusters aux tickets
df["cluster"] = -1
for label, cluster in enumerate(best_clusters):
    df.loc[cluster, "cluster"] = label

# Calcul de la similarité moyenne des descriptions au sein de chaque cluster
sim_desc_moyennes = []
for idx, row in df.iterrows():
    c = row["cluster"]
    indices_cluster = df[df["cluster"] == c].index.tolist()
    if len(indices_cluster) <= 1 or c == -1:
        sim_desc_moyennes.append(0.0)
    else:
        sims = [sim_desc[idx][j] for j in indices_cluster if j != idx]
        sim_desc_moyennes.append(np.mean(sims) if sims else 0.0)

df["SimDesc_Moyenne_Cluster"] = sim_desc_moyennes

# Extraction des mots-clés les plus représentatifs par cluster (sur les titres)
print("Extraction des mots-clés par cluster")

vectorizer_global = TfidfVectorizer(stop_words='french', ngram_range=(1,2))
X_global = vectorizer_global.fit_transform(df["Titre"])
feature_names = np.array(vectorizer_global.get_feature_names_out())

df["Mots_Clés_Cluster"] = ""

for label in df["cluster"].unique():
    if label == -1:
        continue
    indices = df[df["cluster"] == label].index
    cluster_matrix = X_global[indices]
    moyenne_tfidf = cluster_matrix.mean(axis=0).A1
    top_indices = moyenne_tfidf.argsort()[::-1][:20]
    top_mots = feature_names[top_indices]
    df.loc[indices, "Mots_Clés_Cluster"] = " ".join(top_mots)

# Export des résultats finaux
df.to_excel("Tickets_regroupes_motscles_simdesc.xlsx", index=False)
print("---> Export terminé.")

# Export des scores de cohérence par cluster
scores_final = coherence_par_cluster(best_clusters, (sim_titre + sim_desc) / 2)
df_scores = pd.DataFrame(scores_final)
df_scores.to_excel("Coherence_par_cluster.xlsx", index=False)
print("Export des scores de cohérence terminé.")
