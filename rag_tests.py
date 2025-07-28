from retrieval_app import configure_clients, encode_query, build_retrieval_prompt, TOP_N_RESULTS, COLLECTION_NAME
import json


def build_evaluation_prompt(prompt, expected_response, llm_response):
    
    prompt_template = """
{prompt}
Expected Response: {expected_response}
LLM Response: {llm_response}
    """
    return prompt_template.format(prompt=prompt, expected_response=expected_response, llm_response=llm_response)


def generate_llm_response(test_case):

    user_query = test_case['question']
    print("Question: " + user_query + '\n')
    print('Expected Answer: ' + test_case['expected_answer'] + '\n')
    # Set up all external systems
    chroma_client, llm = configure_clients()
    table = chroma_client.get_collection(COLLECTION_NAME)

    # Embed and retrieve
    query_embedding = encode_query(user_query)
    results = table.query(query_embeddings=[query_embedding], n_results=TOP_N_RESULTS)
    context_docs = results["documents"]

    # Generate and display response
    full_prompt = build_retrieval_prompt(context_docs, user_query)
    answer = llm.generate_content(full_prompt).text
    print('LLM Answer: ' + answer + '\n')

    return answer, llm


def main():

    with open("rag_tests.json", "r") as file:
        test_data = json.load(file)

    negative_cases = test_data["negative_test_cases"]
    positive_cases = test_data["positive_test_cases"]

    with open('evaluation_prompts.json', 'r', encoding='utf-8') as f:
        evaluation_prompts = json.load(f)

    print("===========================\n=== Positive Test Cases ===\n===========================\n")
    for i, case in enumerate(positive_cases):
        print('------------------------------------\n\n-------------------------------------')
        print("Case Number: " + str(i+1))
        actual_answer, llm = generate_llm_response(case)
        eval_prompt_formatted = build_evaluation_prompt(evaluation_prompts[0], case['expected_answer'], actual_answer)
        evaluation = llm.generate_content(eval_prompt_formatted).text
        print("Evaluation: " + evaluation + "\n")

    print("===========================\n=== Negative Test Cases ===\n===========================\n")
    for i, case in enumerate(negative_cases):
        print('------------------------------------\n\n-------------------------------------')
        print("Case Number: " + str(i+1))
        actual_answer, llm = generate_llm_response(case)
        eval_prompt_formatted = build_evaluation_prompt(evaluation_prompts[0], case['expected_answer'], actual_answer)
        evaluation = llm.generate_content(eval_prompt_formatted).text
        print("Evaluation: " + evaluation + "\n")


if __name__ == "__main__":
    main()
