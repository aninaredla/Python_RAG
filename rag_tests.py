import chromadb
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
from retrieval_app import configure_clients, load_vector_collection, encode_query, retrieve_relevant_chunks, build_prompt, generate_response
import json


CHROMA_PATH = "./chroma_db"
RESPONSE_LLM = 'gemini-2.5-flash-lite-preview-06-17'
GENAI_API_KEY = "AIzaSyAQ2U0t0yX7kMJuKPWTtcbTaYsHBPN0ELQ"
TOP_N_RESULTS = 5
EMBED_MODEL_ID = "all-MiniLM-L6-v2"


def generate_llm_response(test_case):

    user_query = test_case['question']
    print("Question: " + user_query + '\n')
    print('Expected Answer: ' + test_case['expected_answer'] + '\n')
    # Set up all external systems
    chroma_client, llm = configure_clients()
    table = load_vector_collection(chroma_client)

    # Embed and retrieve
    query_embedding = encode_query(user_query)
    results = retrieve_relevant_chunks(table, query_embedding, 5)
    context_docs = results["documents"]

    # Generate and display response
    full_prompt = build_prompt(context_docs, user_query)
    answer = generate_response(llm, full_prompt)
    print('LLM Answer: ' + answer + '\n')

    return answer, llm


def evaluate_response(actual_answer, llm, test_case):

    EVAL_PROMPT = """
Expected Response: {expected_response}
Actual Response: {actual_response}
---
(Answer with 'true' or 'false') Can the Expected Answer be reasonably inferred using the Actual Answer?
If the actual response contains "I don't know", answer with 'No Evaluation'.
"""

    eval_prompt = EVAL_PROMPT.format(
        expected_response=test_case['expected_answer'], actual_response=actual_answer
    )

    evaluation = llm.generate_content(eval_prompt).text
    print('Correct Response (T/F): ' + evaluation + '\n')


def main():

    with open("rag_tests.json", "r") as file:
        data = json.load(file)

    positive_cases = data["positive_test_cases"]
    negative_cases = data["negative_test_cases"]

    genai.configure(api_key=GENAI_API_KEY)

    print("===========================\n=== Positive Test Cases ===\n===========================\n")
    for i, test_case in enumerate(positive_cases):
        print("Test Number: " + str(i))
        actual_answer, llm = generate_llm_response(test_case)
        evaluate_response(actual_answer, llm, test_case)
    
    print("\n\n===========================\n=== Negative Test Cases ===\n===========================\n")
    for i, test_case in enumerate(negative_cases):
        print("Test Number: " + str(i))
        actual_answer, llm = generate_llm_response(test_case)
        evaluate_response(actual_answer, llm, test_case)


if __name__ == "__main__":
    main()