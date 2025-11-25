import time
from sql_fall_back import sql_fall_back
import pandas as pd
if __name__ == "__main__":
    questions = [
        "How many events occurred in June?",
        "How many events occurred in May?",
        "What is the activity code for paper picking?",
        "What does activity code 290 represent?",
        "What is the address of Balaclava Diamond SE field?",
        "What is the Diamonds: Home to First Base Path - m attribute for Beaconsfield Diamond SW field?",
        "How many events occurred at Angus Park?",
        "How many events occurred at Cambridge Park?",
        "How many parks are larger than 5 hectares?",
        "How many parks are larger than 10 hectares?"
    ]
    results = []
    for query in questions:
        start = time.time()
        result = sql_fall_back(query)
        end = time.time()
        time_taken = end - start
        results.append({"question":query, "result":result["answer_md"], "time_taken":time_taken})
        
        print(f"Time taken to get the answer: {time_taken:.2f} seconds")
        print(result["answer_md"])
    result_df = pd.DataFrame(results)
    result_df.to_excel("sql_fall_back_results.xlsx", index=False)