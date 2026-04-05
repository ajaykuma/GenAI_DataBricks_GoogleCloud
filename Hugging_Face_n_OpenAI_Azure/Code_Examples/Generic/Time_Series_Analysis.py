import pandas as pd
import openai
import os
from openai import AzureOpenAI
import dotenv
from dotenv import load_dotenv

load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("API_KEY"),
    api_version=os.getenv("AZURE_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

deployment_name = os.getenv("AZURE_DEPLOYMENT_NAME")

# ========== LOAD DATA ==========
df = pd.read_csv("https://raw.githubusercontent.com/ajaykuma/Datasets_For_Work/refs/heads/main/opsd_germany_daily.txt")
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Basic preprocessing
df = df.fillna(method='ffill')

# ========== CREATE DATA SUMMARY ==========
def summarize_data(df):
    summary = {
        "shape": df.shape,
        "columns": list(df.columns),
        "date_range": f"{df.index.min()} to {df.index.max()}",
        "stats": df.describe().to_string()
    }
    return summary

# ========== QUERY LLM ==========
def ask_llm(question, df):
    summary = summarize_data(df)

    prompt = f"""
You are a data analyst.
Here is dataset summary:
{summary}

User question:
{question}

Give clear insights and recommendations.
"""

    # response = client.ChatCompletion.create(
    #     model="gpt-4.1",  # or gpt-4 / gpt-5 depending on access
    #     messages=[{"role": "user", "content": prompt}],
    #     temperature=0.3
    # )
    messages = [{"role": "user", "content": prompt}]

    # return response['choices'][0]['message']['content']
    response = client.chat.completions.create(
            model=deployment_name,    # <-- This is the "deployment name" not the raw model name
            messages=messages,
            temperature=0.1,
            top_p=0.8,
            max_tokens=512
        )

    return response.model_dump()  # Return the full response as dict

# ========== SIMPLE CLI APP ==========
def main():
    print("📊 Energy Data Assistant (type 'exit' to quit)\n")

    while True:
        question = input("Ask a question: ")

        if question.lower() == "exit":
            break

        answer = ask_llm(question, df)
        print("\n🤖 Answer:\n", answer)
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()
