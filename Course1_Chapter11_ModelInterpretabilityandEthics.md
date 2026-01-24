11.1 Module Intro

What You'll Learn
Analyze the potential conflicts between the objectives of AI developers and the public interest.
Identify the primary risks and challenges associated with emerging AI technologies.
Define foundation models and their role in modern artificial intelligence.
Articulate the major ethical implications of Generative AI to participate in public discourse.


11.2 Interpretability in AI

Foundation models are frequently described as black boxes because you can only evaluate them by observing their outputs, not by understanding their internal workings. This lack of transparency makes it difficult to trace their reasoning and understand their conclusions.

Several factors contribute to this black box nature:

Immense Scale and Complexity: Models like GPT-4 have trillions of parameters and highly complex architectures, making it nearly impossible for humans to trace their decision-making paths.
Lack of Transparency from Model Providers: Developers often lack access to detailed information about a model's architecture, training data, or specific training processes, which are proprietary.
Probabilistic Nature of Outputs: Models generate probabilities for outputs, and sampling methods can affect the final response. This can lead to inconsistencies and "hallucinations," where the model generates factually incorrect information.
"Emergent" Behaviors: During training, large models can develop unexpected capabilities that were not explicitly programmed. These emergent properties further obscure how the model reaches its results.
Understanding why an AI system makes its decisions—especially biased or incorrect ones—is challenging due to this opacity. This is why interpretability is a crucial skill for AI engineers. It helps in identifying failures, mitigating risks, and ensuring the model is reliable, safe, and aligned with human values.

Illustration depicting two robots engaged in different tasks: on the left, a robot is working on a tax form, indicating it follows a predefined sequence of steps; on the right, another robot is using a grading rubric, showing its focus on structured evaluation.
To increase a model's observability and interpretability, we can adapt methods that we already use to understand complex human processes. Just as humans can sometimes be "black boxes," we've developed tools to add structure and clarity to our work.

Two such methods are:

Follow a predefined sequence of steps: We can ask a model to show its work by breaking down a complex task, like filling out a tax form, into a series of smaller, specific sub-tasks. Each step can then be checked individually for accuracy.
Use a structured rubric: When evaluating something subjective, like the quality of an essay, we can provide the model with a clear rubric. This forces the model to assess the output against specific, predefined scoring criteria, making its evaluation process transparent.
By creating structured processes like forms and rubrics, we provide the structure and reliability needed to make both human and AI reasoning more interpretable.

Foundation models are considered opaque "black boxes" due to their massive scale and complexity, but their reasoning can be made more observable by enforcing structured, step-by-step processes and evaluation criteria.


11.3 
