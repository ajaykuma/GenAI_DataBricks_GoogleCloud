import os
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# Set your Hugging Face API key
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "xxxx"

# Load the Mistral model from Hugging Face
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.2",
    model_kwargs={
        "temperature": 0.5,
        "max_new_tokens": 256,
        "top_p": 0.95,
        "repetition_penalty": 1.1,
    }
)

# === Chain 1: Translate Feedback to English ===
translate_prompt = PromptTemplate(
    input_variables=["feedback"],
    template="""
Translate the following customer feedback into English. Ensure the translation is accurate and retains the original meaning.

Feedback: {feedback}
"""
)
translation_chain = LLMChain(
    llm=llm,
    prompt=translate_prompt,
    output_key="translated_feedback"
)

# === Chain 2: Perform Sentiment Analysis ===
sentiment_prompt = PromptTemplate(
    input_variables=["translated_feedback"],
    template="""
Analyze the sentiment of the following customer feedback. Classify it as Positive, Negative, or Neutral, and provide a brief explanation.

Feedback: {translated_feedback}
"""
)
sentiment_chain = LLMChain(
    llm=llm,
    prompt=sentiment_prompt,
    output_key="sentiment_analysis"
)

# === Chain 3: Generate a Response ===
response_prompt = PromptTemplate(
    input_variables=["sentiment_analysis"],
    template="""
Based on the sentiment analysis below, craft a concise response (20 words or fewer) to address the customer's feedback.

Sentiment Analysis: {sentiment_analysis}
"""
)
response_chain = LLMChain(
    llm=llm,
    prompt=response_prompt,
    output_key="customer_response"
)

# === Combine all chains into a SequentialChain ===
overall_chain = SequentialChain(
    chains=[translation_chain, sentiment_chain, response_chain],
    input_variables=["feedback"],
    output_variables=["translated_feedback", "sentiment_analysis", "customer_response"],
    verbose=True
)

# === Provide customer feedback ===
#customer_feedback = "El producto llegó tarde y estaba dañado. No estoy satisfecho con el servicio."  # Example in Spanish
customer_feedback = "उत्पाद देर से आया और क्षतिग्रस्त था। मैं सेवा से संतुष्ट नहीं हूं।"
result = overall_chain.invoke({"feedback": customer_feedback})

# === Display results ===
print(f"\nOriginal Feedback: {customer_feedback}")
print("\n=== Translated Feedback ===\n", result["translated_feedback"].strip())
print("\n=== Sentiment Analysis ===\n", result["sentiment_analysis"].strip())
print("\n=== Customer Response ===\n", result["customer_response"].strip())