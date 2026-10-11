# Résultats de l'évaluation

30 questions, bot `qwen3:4b-instruct-2507-q4_K_M`, juge RAGAS : `qwen3:4b-instruct-2507-q4_K_M` (même modèle que le bot).

| Mesure | recursive | structure |
|---|---|---|
| Recherche : hit rate@6 | 82 % | 91 % |
| Recherche : MRR | 0,62 | 0,71 |
| RAGAS : faithfulness | 0,66 | 0,71 |
| RAGAS : answer_relevancy | 0,63 | 0,64 |
| RAGAS : context_precision | 0,47 | 0,41 |
| RAGAS : context_recall | 0,61 | 0,73 |
| Règles métier respectées | 73 % | 93 % |
| Temps de réponse moyen (s) | 10,14 | 7,59 |

## Règles métier non respectées

- `recursive` **allerg-01** : liste officielle affichée
- `recursive` **allerg-02** : liste officielle affichée
- `recursive` **allerg-03** : liste officielle affichée
- `recursive` **allerg-05** : liste officielle affichée
- `recursive` **allerg-06** : liste officielle affichée
- `recursive` **produit-02** : contient « 19,50 »
- `recursive` **produit-04** : contient « croissants »
- `recursive` **produit-06** : contient « gressinis »
- `structure` **allerg-03** : contient « poissons » ; liste officielle affichée
- `structure` **produit-04** : contient « croissants »

## Source attendue absente des extraits

- `recursive` **delai-04**
- `recursive` **produit-02**
- `recursive` **produit-04**
- `recursive` **produit-06**
- `structure` **delai-04**
- `structure` **produit-04**
