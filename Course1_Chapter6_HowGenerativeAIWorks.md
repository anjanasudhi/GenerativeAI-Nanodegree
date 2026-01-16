6.3 - GenAT architectures Apply & Explore

arXiv — Attention Is All You Need (Vaswani et al., 2017)(opens in a new tab) — The original Transformer paper that introduced the attention-only architecture underlying modern LLMs.
Scale AI — Diffusion Models: A Practical Guide(opens in a new tab) — Practical, developer-oriented explanation of how diffusion models add noise and learn to denoise to generate images.
DiffusionFlow (blog) Diffusion Meets Flow Matching: Two Sides of the Same Coin(opens in a new tab) — Tutorial/blog post showing the connection and equivalence between diffusion models and Gaussian flow-matching formulations.
Clemson / PyTorch LLM tutorial — Small Language Models: an introduction to autoregressive (causal) modeling(opens in a new tab) Practical notes on autoregressive (causal) language modeling used by decoder-only LLMs (predict next token conditioned on previous tokens).


6.4 Training GenAI MOdels


Training a generative AI model, like a text-based Transformer, begins with a process called pre-training. This stage uses a vast and diverse dataset containing content like books, articles, encyclopedias (e.g., Wikipedia), online forums, and code repositories from sources like GitHub.

The core task during pre-training is next-token prediction. The model is given a snippet of text and must guess the next word. For example, given the phrase "To be or not to...", the model should predict "be". If the model's prediction is correct, its internal parameters (weights) are adjusted to reinforce that choice. If it's incorrect, the weights are adjusted to discourage that choice in the future. This initial step gives the model a foundational understanding of language.

Pre-training involves teaching a generative AI model to complete text from a diverse array of sources, such as Shakespeare.
Pre-training generative AI models.

After pre-training, the model undergoes fine-tuning, which is also increasingly referred to as post-training. This stage steers the pre-trained model to perform specific tasks or to better understand a particular domain by further adjusting its weights. Fine-tuning typically occurs in two main phases.

The first phase is Supervised Fine Tuning (SFT). This is a form of supervised learning that uses a high-quality, labeled dataset to teach the model to mimic "gold-standard" responses on a token-by-token basis. SFT is crucial for teaching a model to follow instructions (like "translate this to French") and to engage in chat-style interactions. Without it, a model might simply continue generating text from its pre-training data rather than responding to a user's prompt.

The second phase is Reinforcement Fine Tuning, sometimes called preference fine tuning. Instead of optimizing for next-token prediction, this method optimizes for the overall quality of a generated sequence. It uses feedback—either from humans or automated, rubric-based criteria—to learn which responses are "preferred." For instance, it can be trained on a dataset of preferred versus rejected responses. This method generally requires less data than SFT and is key to improving a model's reasoning ability.

Supervised fine-tuning encourages a model to mimic a data set exactly token by token, while reinforcement fine-tuning encourages the same model to complete entire sequences that are preferred over others.
Supervised Fine-Tuning vs Reinforcement Fine-Tuning

When these training stages are combined, a remarkable capability emerges: reasoning. For a Large Language Model (LLM), reasoning is the ability to break down a complex problem into smaller steps and arrive at a correct conclusion, similar to mathematical or analytical thinking.

This can be understood through the analogy of System 1 and System 2 thinking from human psychology.

System 1 thinking is rapid, intuitive, and automatic. This is akin to a basic LLM performing next-token prediction to give an immediate response.

System 2 thinking is slow, deliberate, and analytical. This mirrors an advanced LLM using a technique like chain of thought, where it breaks a complex task into intermediate steps to find a solution.

System 1 thinking is rapid, intuitive, and automatic, while System 2 thinking is slow, deliberate, and analytical.
System 1 versus System 2 Thinking


6.5 Training GenAI Models: Apply and Explore

Hugging Face Blog - Illustrating Reinforcement Learning from Human Feedback (RLHF)(opens in a new tab) A practical overview of RLHF describing reward modeling, human preference data, and how RL is used to align LLM outputs with human judgments.
arXiv — Chain-of-Thought Prompting Elicits Reasoning in Large Language Models(opens in a new tab) The original paper showing how providing intermediate reasoning steps (chain-of-thought) in prompts substantially improves complex reasoning performance in large models.
Hugging Face LLM Course — Supervised Fine-Tuning (SFT)(opens in a new tab) A hands-on tutorial-style chapter explaining SFT: preparing labeled data, training steps, and best practices for adapting pretrained LLMs to task-specific behavior.
fs.blog — Daniel Kahneman Explains The Machinery of Thought (System 1 & System 2)(opens in a new tab) A readable essay that summarizes Kahneman’s System 1 (fast/intuitive) vs System 2 (slow/deliberative) framework and its implications for decision-making.
Advances in reinforcement fine-tuning have significantly improved this "System 2" capability, enabling even smaller models to reason about complex topics more effectively than some larger models.

A multi-stage training process—combining broad pre-training with specialized supervised and reinforcement fine-tuning—is what allows generative models to move beyond simple pattern matching to complex reasoning.




