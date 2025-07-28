# Retrieval-Augmented Generation (RAG) Evaluation Framework

## Project Goal
- Implement a modular framework to test different retrieval prompts and evaluation prompts for RAG-style question answering.
- Evaluate the effectiveness of RAG pipelines using custom test cases and semantic-based scoring.

## Dataset Summary
The evaluation was conducted using 12 custom test cases. Each test case includes:
- A **question** to be answered,
- A set of **retrieved documents** simulating context retrieval,
- An **expected answer** that reflects the ground truth,
- A **generated answer** from a language model.

The RAG framework is tested using combinations of:
- **4 Retrieval Prompts** that guide how the context is combined with the question,
- **3 Evaluation Prompts** to determine whether the generated answer meets specific criteria (semantic similarity, inference capability, average person understanding).

## Skills Used
Developed an evaluation pipeline that tests multiple prompt configurations using Python. Implemented functions to generate answers, build prompts dynamically, and score responses using evaluation LLM prompts. The goal was to simulate realistic retrieval scenarios and assess the value of structured prompt engineering.

## Technical Skills
Utilized:
- `OpenAI` APIs for answer and evaluation generation,
- Python functions to construct prompts dynamically,
- JSON and dictionary structures for managing test metadata,
- Programmatic loops to iterate across different retrieval/evaluation combinations.

## Real-World Impact
This project provides a foundation for:
- Testing prompt engineering in RAG systems at scale,
- Understanding how retrieval phrasing affects answer quality,
- Automating qualitative answer evaluation using LLMs.

It also highlights the challenges in automatically evaluating model outputs, especially when dealing with nuanced semantics or ambiguous phrasing.

## Visualization Descriptions (optional)
While this iteration focused on textual outputs, future versions may include:
- Bar plots comparing average scores across retrieval prompts,
- Confusion matrices or heatmaps to track false negatives in semantic matching,
- Prompt-level analytics summarizing which evaluation prompts were too lenient or strict.
