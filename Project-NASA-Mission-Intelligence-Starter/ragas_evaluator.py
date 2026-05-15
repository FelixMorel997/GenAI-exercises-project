from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional
import os

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas import EvaluationDataset
    from ragas.metrics import ResponseRelevancy, Faithfulness
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    api_key = os.getenv("OPENAI_API_KEY", "")
    if api_key.startswith("voc"):
        base_url = "https://openai.vocareum.com/v1"
    else:
        base_url = None

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
        api_key=api_key,
        base_url=base_url,
    )
    evaluator_llm = LangchainLLMWrapper(llm)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=api_key,
        base_url=base_url,
    )
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings)
    ]

    sample = SingleTurnSample(
        user_input=question,
        response=answer,
        retrieved_contexts=contexts,
    )

    dataset = EvaluationDataset.from_list([sample.to_dict()])
    result = evaluate(
        dataset,
        metrics=metrics,
    )

    df = result.to_pandas()
    return df.iloc[0].to_dict()
