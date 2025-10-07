## Setup
```
pip install openai
pip install sentence-transformers
```
## Bilingual Evaluation
In ``eval_bilingual.py``, we use SentenceTransformer to evaluate the performance of bilingual interpretation. Please change the ``src`` to the path of the JSON file.

## Intent Evaluation
In ``eval_intent``, we provide an example code of using DeepSeek to evaluate the quality of intent analysis. You should have your own api-key, and you can also change the type of model freely.