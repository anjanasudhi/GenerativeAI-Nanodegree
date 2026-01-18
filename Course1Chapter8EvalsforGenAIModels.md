8.1 Module Summary

What You'll Learn
Apply various metrics like Exact Match, ROUGE, and Semantic Similarity to evaluate text outputs.
Assess the functional correctness of generated code using unit tests and the Pass@k metric.
Implement the "LLM-as-a-Judge" pattern by creating a rubric to evaluate subjective model outputs.
Select the appropriate evaluation technique based on the specific Generative AI task.

8.2 Demo:

Exact Match (EM)
Our first technique is Exact Match. This is the simplest and strictest evaluation metric. It checks if the model's output is perfectly identical to the reference answer. 
It's most useful for tasks that have a single, clear correct answer, like a multiple-choice question.

In this example, we compare a list of predicted fruit names against the correct labels. We define a simple normalize function to make the text lowercase and remove extra 
whitespace before comparing.


Lexical Similarity (ROUGE)
Exact Match is too strict when a correct answer can be phrased in different ways. For that, we can use ROUGE (Recall-Oriented Understudy for Gisting Evaluation). Instead of
requiring a perfect match, ROUGE measures the overlap of words or n-grams between the model's prediction and the reference label. It's very commonly used for evaluating 
text summarization.

Here, we compare two sentences with similar words but different structures. We use the evaluate library to load the ROUGE metric. The output shows two scores:
ROUGE-1 measures the overlap of individual words.
ROUGE-L measures the longest common sequence of words.


Semantic Similarity
What if two sentences mean the same thing but use completely different words? That's where Semantic Similarity comes in. This technique converts sentences into numerical
vectors called embeddings and then measures the similarity between them, typically using cosine similarity. A score close to 1.0 means the sentences are very similar in meaning.

First, we load a pre-trained model good at creating sentence embeddings. We then generate embeddings for our predictions and labels and calculate the cosine similarity for
each pair.


Functional Correctness
For tasks involving code generation, we need to know if the code actually works. Functional Correctness evaluates this by running the generated code against a set of unit tests. 
The final score is the proportion of tests that pass.

In this example, we have a Python function that is supposed to reverse and capitalize a string but contains a bug: it fails if the input contains a number. We test it with 
three inputs.


Pass@k
Sometimes we ask a model to generate multiple (k) possible answers for a single problem. The Pass@k metric measures if at least one of these k attempts is correct. If any 
sample is correct, the entire set is considered a "pass" with a score of 1.

Here, we imagine a model generated four possible answers when asked to name a primary color. Our function checks if the correct label, "blue," is present in the list 
of samples.


LLM-as-a-Judge
For complex and subjective tasks, like judging creativity or helpfulness, we can use another powerful LLM to act as a judge. We provide this judge with the model's 
prediction, a reference answer, and a detailed rubric. The judge then provides a score and its reasoning.

First, we define a clear rubric for scoring animal predictions. Our judge function then evaluates three different test cases based on this rubric.


