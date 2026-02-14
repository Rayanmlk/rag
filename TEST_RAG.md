# Comment tester le RAG et interpréter les résultats

---

## 1. Une question qui montre une grande réussite

La base contient des résumés de 5 papers : **Transformer**, **BERT**, **GPT-3**, **RAG**, **Sentence-BERT**.

**Question recommandée (réponse très explicite dans les docs) :**

```
What are the two pre-training objectives of BERT?
```

**Réponse attendue :** Masked Language Modeling (MLM) et Next Sentence Prediction (NSP). Cette phrase est quasi textuelle dans le document BERT.

**Autres questions qui marchent très bien :**
- `How many parameters does GPT-3 have?` → 175 billion
- `What does RAG combine?` → parametric memory (LLM) and non-parametric memory (retrieval)
- `What is the BLEU score of the Transformer on WMT 2014 English-German?` → 28.4
- `How does Sentence-BERT reduce semantic search time for 10,000 sentences?` → from 65 hours to 5 seconds (pre-computed embeddings)

---

## 3. Comment interpréter la réussite

Après avoir cliqué sur **Envoyer**, regarde :

| Élément | Bon signe | À surveiller |
|--------|-----------|--------------|
| **Réponse** | Réponse courte, factuelle, qui reprend les infos des papers. | Réponse vague, « je ne sais pas », ou hors-sujet. |
| **Sources** | Les tags montrent des sources pertinentes (ex. « Devlin et al., 2018 » pour BERT). | Sources sans lien avec la question. |
| **Scores de similarité** | Premier passage > 0,7 (70 %) ; plusieurs passages au-dessus de 0,5. | Scores bas (< 0,5) → la question ne matche pas bien le corpus. |
| **Temps** | Recherche (retrieval) < ~100 ms ; total raisonnable selon le LLM. | Très long ou erreur timeout. |

**En résumé :**  
Une **grande réussite** = bonne réponse factuelle + sources cohérentes + similarités élevées sur les premiers passages.  
Une **réussite moyenne** = bonne idée mais réponse incomplète ou 1–2 sources utiles.  
Un **échec** = pas de réponse utile ou similarités faibles (question hors du domaine des 5 papers).

**Note :** Si le LLM est en « fallback » (sans clé API), tu vois un extrait du meilleur passage au lieu d’une vraie réponse LLM ; les scores de similarité et les sources restent le reflet direct de la qualité du retrieval.
