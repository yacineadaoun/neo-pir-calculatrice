# Interprétation NEO PI-R (Cabinet Pro)

> Objectif : fournir une lecture clinique structurée des résultats NEO PI-R à partir des **scores bruts**, puis des **scores normés** (T, percentiles, niveaux), en respectant les **règles de validité du protocole**.

## 1) Rappels de structure

Le NEO PI-R mesure 5 grands domaines (“Big Five”) et 30 facettes (6 par domaine). :contentReference[oaicite:2]{index=2}

- **N** : Névrosisme (N1..N6)
- **E** : Extraversion (E1..E6)
- **O** : Ouverture (O1..O6)
- **A** : Agréabilité (A1..A6)
- **C** : Conscience (C1..C6)

Les facettes sont la lecture la plus fine ; le domaine est une synthèse.

---

## 2) Prérequis : validité du protocole

Avant toute interprétation :
1. Vérifier la **qualité de complétion** (omissions / blancs).
2. Vérifier la **cohérence** et les **styles de réponse** (réponses uniformes, acquiescement, etc.).
3. Si le protocole est jugé invalide : **ne pas interpréter** (ou interpréter avec une prudence majeure et le documenter).

Le manuel comporte une section dédiée aux **indicateurs de validité des réponses**. :contentReference[oaicite:3]{index=3}

> Recommandation “Cabinet” : afficher un statut (VALIDE / EN COURS / INVALIDE) + les raisons (ex. trop de blancs, trop de réponses neutres, etc.), et bloquer l’édition “rapport clinique” si invalide.

---

## 3) Scores : brut → normé

### 3.1 Scores bruts
Le score brut se calcule par addition des points selon la clé de correction (FD/D/N/A/FA), avec inversion pour certains items (selon clé).  
- Domaines : somme des facettes du domaine.
- Facettes : somme des items rattachés à la facette.

### 3.2 Scores normés (T-scores / percentiles)
La lecture clinique se fait idéalement en **scores normés**, distincts selon :
- sexe (au minimum homme/femme)  
- parfois âge/niveau (selon normes disponibles)

Tu disposes de la **feuille de profil officielle** (H/F) : elle sert de référence pour positionner les scores sur des zones (très faible → très élevé). :contentReference[oaicite:4]{index=4}

> Implémentation recommandée : stocker les normes en CSV (`data/norms_f.csv`, `data/norms_m.csv`) et convertir automatiquement `raw → T`.

---

## 4) Principes d’interprétation (cadre clinique)

### 4.1 Interprétation par domaine
- Un **score élevé** indique une tendance forte du trait.
- Un **score faible** indique l’inverse (ou un trait peu exprimé).
- La signification exacte dépend du **profil global**, du **contexte clinique**, et de la **validité**.

Toujours interpréter :
1) Domaine (vue globale)  
2) puis facettes (profil interne du domaine)  
3) et enfin cohérence inter-domaines.

### 4.2 “Très élevé / Élevé / Moyen / Faible / Très faible”
Ces catégories doivent venir des **normes** (T ou percentiles).
- Exemple (si tu utilises T-scores) : tu peux définir des bandes paramétrables (ex. T≥65 élevé, T≤35 faible), mais la référence doit rester les normes retenues.

---

## 5) Interprétation par facettes (résumé opérationnel)

> Les libellés ci-dessous donnent une lecture “cabinet” synthétique. La formulation finale doit rester prudente (“tendance à…”, “peut indiquer…”), et toujours contextualisée.

### N — Névrosisme
- **N1 Anxiété** : inquiétude, tension, anticipation négative.
- **N2 Hostilité** : irritabilité, colère, ressentiment.
- **N3 Dépression** : humeur triste, découragement, pessimisme.
- **N4 Timidité** : gêne sociale, inhibition.
- **N5 Impulsivité** : difficulté à résister aux envies/urgences.
- **N6 Vulnérabilité** : sensibilité au stress, difficulté à faire face.

### E — Extraversion
- **E1 Chaleur** : proximité, cordialité.
- **E2 Grégarité** : goût des groupes, sociabilité.
- **E3 Affirmation** : assertivité, leadership.
- **E4 Activité** : énergie, rythme élevé.
- **E5 Excitation** : recherche de sensations.
- **E6 Émotions+** : enthousiasme, joie.

### O — Ouverture
- **O1 Imagination** : créativité, monde intérieur.
- **O2 Esthétique** : sensibilité artistique.
- **O3 Sentiments** : profondeur émotionnelle.
- **O4 Actions** : ouverture au changement/expériences.
- **O5 Idées** : curiosité intellectuelle.
- **O6 Valeurs** : tolérance, remise en question.

### A — Agréabilité
- **A1 Confiance** : présomption d’honnêteté d’autrui.
- **A2 Franchise** : sincérité, transparence.
- **A3 Altruisme** : empathie, aide.
- **A4 Compliance** : coopération, évitement du conflit.
- **A5 Modestie** : humilité.
- **A6 Tendresse** : sensibilité, compassion.

### C — Conscience
- **C1 Compétence** : efficacité perçue, capacité à faire face.
- **C2 Ordre** : organisation, structure.
- **C3 Devoir** : respect des règles, morale.
- **C4 Effort** : persévérance, ambition.
- **C5 Autodiscipline** : constance, contrôle.
- **C6 Délibération** : prudence, réflexion avant action.

---

## 6) Rédaction clinique (modèle)

### 6.1 Résumé profil
- Validité protocole : VALIDE/INVALIDE + raisons.
- Domaines saillants : (ex. N élevé, C faible…)
- Facettes qui expliquent le domaine : (ex. N élevé surtout via N1+N3)

### 6.2 Hypothèses prudentes
Toujours employer :
- “peut être compatible avec…”
- “suggère une tendance à…”
- “à confronter avec l’entretien / l’anamnèse / les observations”

### 6.3 Limites
- auto-questionnaire : biais de désirabilité / compréhension / contexte
- normes : dépendance à l’échantillon de référence
- protocole invalide : prudence majeure

---

## 7) Références internes du projet
- Manuel professionnel (sections normes, validité, interprétation). :contentReference[oaicite:5]{index=5}
- Feuille de profil officielle (H/F) pour lecture normative. :contentReference[oaicite:6]{index=6}
