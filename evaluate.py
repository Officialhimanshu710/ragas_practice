from chain import rag_chain, retriever, llm, embeddings

import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

questions = [
    "What is the importance of pH in everyday life?",
    "What happens when an acid reacts with a base?"
]

ground_truth = [
    "pH is important for digestion in our stomach, tooth decay prevention, and soil fertility for plants.",
    "When an acid reacts with a base, they neutralize each other to form salt and water."
]

answers = []
contexts = []

print("Now evaluating the ragas metrics...")

for q in questions:
    docs = retriever.invoke(q)
    contexts.append([doc.page_content for doc in docs])
    ans = rag_chain.invoke(q)
    answers.append(ans)

# Build HuggingFace Dataset
data = {
    "question": questions,
    "ground_truths": [[gt] for gt in ground_truth],
    "reference": ground_truth,
    "answer": answers,
    "contexts": contexts
}
dataset = Dataset.from_dict(data)

# Wrap LLM and Embeddings for Ragas
ragas_llm = LangchainLLMWrapper(llm)
ragas_embedding = LangchainEmbeddingsWrapper(embeddings)

# Run evaluation
result = evaluate(
    dataset=dataset,
    metrics=[
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ],
    llm=ragas_llm,
    embeddings=ragas_embedding
)

print("------ Final Scores -------")
df = result.to_pandas()
print(df[["faithfulness", "answer_relevancy", "context_precision", "context_recall"]])

# Save to CSV
df.to_csv("ragas_results.csv", index=False)
print("Results saved to ragas_results.csv")
