# Python Retrieval-Augmented Generation (RAG) Program: Chatbot and Evaluation

## Project Goal
- Implement a retrieval augmented generation program for a chatbot for question-answering tasks.
- Evaluate the effectiveness of the program using custom test cases and semantic-based scoring.

## Dataset Summary
- The input data is a research paper in PDF format, which serves as the knowledge base of the program (to be expanded later).
- The test cases involve questions asking about different parts of the paper.

The RAG framework is tested using **3 Evaluation Prompts** to determine whether the generated answer meets specific criteria (semantic similarity, inference capability, average person understanding).

## Skills Used
Developed an evaluation pipeline that tests multiple prompt configurations using Python. Implemented functions to generate answers, build prompts dynamically, and score responses using evaluation LLM prompts. The goal was to simulate realistic retrieval scenarios and assess the value of structured prompt engineering.

## Technical Skills
Utilized:
- Google Gemini LLMs for answer and evaluation generation (to be changed),
- Python functions to construct prompts dynamically,
- JSON and dictionary structures for managing tests.

## Real-World Impact
This project provides a foundation for:
- Testing prompt engineering in RAG systems at scale,
- Understanding how retrieval phrasing and query specificity affect answer quality,
- Automating qualitative answer evaluation using LLMs.
- Highlights the challenges in automatically evaluating model outputs, especially when dealing with nuanced semantics or ambiguous phrasing.
