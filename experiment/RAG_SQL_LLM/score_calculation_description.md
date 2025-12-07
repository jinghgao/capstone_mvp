
The evaluation process uses RAG + SQL pipeline. The main goal of this evaluation is accuracy. 
This pipeline is designed to handle field dimension comparison tasks. e.g. Compare Balaclava Diamond SE field dimensions to standard softball female U17 field requirements","Compare Adanac Field NW field dimensions to standard soccer U8 field requirements","What are the dimension difference between Balaclava Field Oval and standard U12 soccer field?"

## The input dataset - 10 in total:
For each input sample, it has: 
- query: a natural-language comparison prompt (string)
- retrieval_ground_truth: the expected context/standard
- answer_ground_truth: expected structured answer (dict with boolean keys like "Home to Pitchers Plate" and numeric diff keys like "Pitchers Plate Difference")
- retrieved_field_data: the measured field values used for comparison

## Process 
per sample, for items dataset[10:20]

- Retrieval: call rag_retrieve(query) to get top KB hits; extract the hit texts.
- Reasoning: call llm_reason(retrieved_texts, query, sql=retrieved_field_data) with tetrieved text from RAG, field dimension from SQL query and natural languange query to llm

- Compare parsed model output to answer_ground_truth:
    - Boolean keys (e.g., "Home to Pitchers Plate"): if there is such key in ground_truth sample, check/denominator plus 1. Award 1 point if the model has the key and the boolean value equals the expected boolean.
    - Numeric diff keys (e.g., "Pitchers Plate Difference"):
        - If there is a Numeric diff key, check/denominator plus 1.
        - If expected is None and model returns None → award 1 point.
        - If expected is a number and model returns a number within ±1.0 meters of expected → award 1 point.

- Score calculation:
    - score = number of passed checks (sum of points).
    - checks = number of expected keys evaluated for that sample.
    - normalized score = score / max(checks, 1) (so in [0.0, 1.0]).
