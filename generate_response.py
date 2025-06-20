import os
import openai

from vectors_search import search_similar_chunks

# Load your OpenAI key

openai.api_key = os.getenv("API")

def build_prompt(user_query, context_chunks):
    context_text = "\n\n".join(context_chunks)
    prompt = f"""
You are a helpful and ethical FAQ assistant for a healthcare company.

Answer the question below using only the context provided.

Context:
{context_text}

Question: {user_query}
Answer (mention source if possible):
"""
    return prompt.strip()

def generate_response(user_query):
    results = search_similar_chunks(user_query)
    context_chunks = results["documents"][0]

    prompt = build_prompt(user_query, context_chunks)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # You can change to gpt-4 if needed
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4,
            max_tokens=300
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        return f"❌ Error: {str(e)}"

if __name__ == "__main__":
    query = input("Ask a question: ")
    answer = generate_response(query)
    print("\n🔹 Response:\n")
    print(answer)
