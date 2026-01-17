7.2 Challenges in GenAI Model Evaluation

Key challenges in GenAI model evaluation include:

Open-Ended Outputs: Traditional ML tasks often have a single, correct answer. GenAI models, however, produce a wide range of plausible, open-ended responses. For example, when evaluating a generated poem, there is no single "correct" version. This makes it impossible to create an exhaustive list of expected outputs for comparison.

Inherent Complexity of Tasks: GenAI tackles inherently complex problems such as generating mathematical proofs, writing code, or creating fiction. Evaluating these sophisticated outputs often requires deep human expertise, nuanced reasoning, and thorough fact-checking, making the process highly time-consuming.

Probabilistic Nature: The models generate responses based on probabilities, which can lead to inconsistency across different runs even with the same input. A thorough evaluation may require running the model multiple times to assess the range of potential outputs, which drives up computational costs.

Underinvestment in Evaluation: Despite its importance, systematic evaluation can be under-resourced when time and budget are tight. This often leads teams to rely on ad-hoc methods like word-of-mouth or simply "eyeballing" results, which increases the risk of deploying a flawed or unreliable model.

Prompt Sensitivity: Model performance can be highly sensitive to minor variations in input prompts. Small changes in wording can drastically alter the output, making consistent and repeatable evaluation difficult. This problem is amplified in systems where the output of one GenAI model becomes the input for another.

Despite these difficulties, it is still possible to utilize specific methods to effectively evaluate GenAI models.

Evaluating GenAI models is uniquely challenging due to their open-ended outputs, probabilistic nature, and sensitivity to prompts, requiring methods beyond traditional ML accuracy metrics.


7.3 Evaluation Methods for GenAI Models


Exact Evaluation is a method that yields unambiguous, deterministic judgments about a model's performance. This contrasts sharply with subjective evaluations, like grading an essay, which can vary between evaluators. While this approach is perfect for tasks with a single clear, correct answer, it can also be adapted for more open-ended responses.

There are two primary approaches to exact evaluation:

Functional Correctness: This evaluates whether a system performs its intended functionality. For example, if a model is asked to book a restaurant reservation, functional correctness assesses if the reservation was accurately made. In coding tasks, this translates to execution accuracy—whether the generated code runs and produces the expected output. While functional correctness is the ultimate metric for an application's success, it isn't always straightforward to measure and can be challenging to automate.

Similarity Measurements Against Reference Data: This method involves comparing a model's outputs to pre-existing ground truth or canonical responses (reference data). The effectiveness of this approach depends on the quality of the reference data, which can be costly and time-consuming to generate.

There are three main ways to measure this similarity:

Exact Match: A binary (yes/no) measure suitable for simple questions with definitive answers, like trivia or basic math problems.
Lexical Similarity: Measures the overlap of words or tokens between the generated output and the reference text. Techniques include edit distance or n-gram overlap metrics like BLEU and ROUGE. A drawback is that high lexical similarity doesn't always guarantee a better or semantically equivalent response.
Semantic Similarity: This is a continuous measure that determines if two texts convey the same meaning, regardless of the exact words used. It's typically measured by comparing text embeddings using metrics like cosine similarity.
AI as a Judge

The workflow is as follows:

A Prompt is sent to the Language Model (LLM) being tested.
The LLM generates a Response.
A new Evaluation Prompt is created, containing the original prompt and the response, along with specific instructions or a question for the evaluator model (e.g., "Is the response contained within the prompt?").
This evaluation prompt is sent to an Evaluation LLM (the "judge").
The judge provides an Evaluation Result (e.g., "Yes/No").
