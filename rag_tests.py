import time
from retrieval_app import configure_clients, encode_query, build_retrieval_prompt, TOP_N_RESULTS, COLLECTION_NAME
import json
from eval_prompts import EVAL_PROMPTS


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
    print('Expected Answer: ' + test_case['reference'] + '\n')
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

    return answer, context_docs, llm


def main():

    # with open("driving_questions.json", "r") as file:
    #     test_data = json.load(file)
    
    # results = []
    # LLM_answers = []
    # # test_instance = test_data[3]
    # for i, test_instance in enumerate(test_data):
    #     print(f"\n\nTEST NUMBER: {str(i+1)}\n\n")

    #     question = test_instance['question']
    #     reference = test_instance['reference']

    #     start = time.perf_counter()
    #     answer, context_docs, llm = generate_llm_response(test_instance)
    #     LLM_answers.append(answer)

    #     correctness_prompt = EVAL_PROMPTS['correctness_prompt'].format(
    #         input=question, 
    #         output=answer, 
    #         reference_output=reference
    #     )
    #     correctness_evaluation = llm.generate_content(correctness_prompt).text
    #     print("\n\nCorrectness Evaluation:\n" + correctness_evaluation)

    #     groundedness_prompt = EVAL_PROMPTS['groundedness_prompt'].format(
    #         context=context_docs, 
    #         output=answer
    #     )
    #     groundedness_evaluation = llm.generate_content(groundedness_prompt).text
    #     print("\n\nGroundedness Evaluation:\n" + groundedness_evaluation)

    #     helpfulness_prompt = EVAL_PROMPTS['helpfulness_prompt'].format(
    #         input=question, 
    #         output=answer
    #     )
    #     helpfulness_evaluation = llm.generate_content(helpfulness_prompt).text
    #     print("\n\nHelpfulness Evaluation:\n" + helpfulness_evaluation)

    #     retrieval_relevance_prompt = EVAL_PROMPTS['retrieval_relevance_prompt'].format(
    #         input=question, 
    #         context=context_docs
    #     )
    #     retrieval_relevance_evaluation = llm.generate_content(retrieval_relevance_prompt).text
    #     print("\n\nRetrieval Relevance Evaluation:\n" + retrieval_relevance_evaluation)

    #     # Append to results list
    #     results.append({
    #         "question": question,
    #         "reference": reference,
    #         "llm_answer": answer,
    #         "correctness_evaluation": correctness_evaluation,
    #         "groundedness_evaluation": groundedness_evaluation,
    #         "helpfulness_evaluation": helpfulness_evaluation,
    #         "retrieval_relevance_evaluation": retrieval_relevance_evaluation
    #     })

    #     finish = time.perf_counter()
        
    #     # print(f"\n\nResponse generation took {str(middle-start)} seconds and response evaluation took {str(finish-middle)} seconds.\n\n")
    #     if ((30.0 - (finish-start)) > 0):
    #         time.sleep((30.0 - (finish-start) + 1.0))
    #         print(f"\n\nSleeping for {str((30.0 - (finish-start)))} seconds to be within rate limits.\n\n")

    # with open("evaluation_results_4.json", "w", encoding="utf-8") as f:
    #     json.dump(results, f, indent=4, ensure_ascii=False)

    # with open("LLM_answers_4.json", "r", encoding="utf-8") as f:
    #     json.dump(LLM_answers, f, indent=4, ensure_ascii=False)

    with open("evaluation_results_4.json", "r", encoding="utf-8") as file:
        eval_results_2 = json.load(file)

    for i,elem in enumerate(eval_results_2[0:12]):
        print("\n\nTest Number " + str(i+1) + ":")
        for key in elem.keys():
            print(key)
            print(elem[key])
            print("\n\n")

if __name__ == "__main__":
    main()