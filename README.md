# Loop Solver — Mini POC

Pattern Mining • Machine Learning • Interaction • Contraintes CP

---

## Objectif

* apprendre des contraintes à partir des données
* les transformer en décisions
* permettre l’interaction et l’affinage par l’utilisateur

> C’est un **Proof of Concept**, pas un outil final.

---

## Architecture

```
LoopSolver/
│
├── data/
│   ├── patients.csv
│   └── tmp_new.csv
│
├── src/
│   ├── data_loader.py
│   ├── pattern_mining.py
│   ├── model.py
│   ├── constraints.py
│   ├── feedback.py
│   ├── utils.py
│   └── app.py
│
├── templates/
├── static/
│
├── requirements.txt
├── main.py
└── README.md
```

---

## Dataset (exemple)

| age | urgency | complexity | delay | priority |
| --- | ------- | ---------- | ----- | -------- |
| 55  | high    | low        | 75    | 1        |
| 34  | low     | low        | 5     | 0        |
| 79  | medium  | high       | 15    | 1        |

* `priority = 1` → prioritaire
* `priority = 0` → moins urgent

---

## Pipeline général

### 1 Préparation & encodage

* nettoyage
* encodage des catégories
* séparation features / labels

### 2 Pattern Mining

Extraction de règles candidates :

```
delay > 60 -> priority = 1
urgency_high -> priority = 1
```

Mesures calculées :

* **Support**
* **Confidence**

### 3 Machine Learning (Decision Tree)

Apprentissage d’un modèle supervisé et affichage des règles apprises.

### 4 Interaction utilisateur (boucle)

Depuis l’interface web :

* ajouter des nouveaux patients
* trier selon priorité
* prédire la priorité
* corriger / ajouter au dataset
* proposer une règle
* tester sa satisfaction

### 5 Contraintes CP (OR-Tools)

Vérifier la cohérence de certaines règles sur un ensemble de cas.

---

## Installation

### 1 Créer l’environnement

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Linux / Mac
```

### 2 Installer les dépendances

```bash
pip install -r requirements.txt
```

### 3 Lancer

```bash
python main.py
```

Puis ouvrir :

```
http://127.0.0.1:5000
```

---

## Fonctionnalités

✔ Visualiser les patterns
Support + Confidence des règles extraites.

✔ Ajouter de nouveaux patients
Chargement CSV puis prédiction automatique.

✔ Feedback utilisateur
Possibilité d’ajouter de nouveaux cas et ré-entraîner.

✔ Tester des règles proposées
Exemple :

```
delay>60 -> priority=1
```

Le système calcule :

* satisfaction
* cohérence
* validation

---

## Concepts

**Support**
Proportion de lignes où condition + conclusion sont vraies.

**Confidence**
Parmi les cas où la condition est vraie, combien respectent aussi la conclusion.

**Satisfaction**
Part globale du dataset respectant la règle.
