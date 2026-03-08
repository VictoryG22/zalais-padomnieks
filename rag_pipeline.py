from openai import OpenAI

client = OpenAI()

def rag_answer(question, history, vectorstore):

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 12, "fetch_k": 50}
    )

    docs = retriever.invoke(question)

    context_blocks = []
    sources = []

    for d in docs:
        src = d.metadata.get("source", "Unknown")
        page = d.metadata.get("page", None)

        sources.append({"source": src, "page": page})

        header = f"[SOURCE: {src}" + (f", page {page}]" if page else "]")
        context_blocks.append(header + "\n" + d.page_content)

    context_text = "\n\n---\n\n".join(context_blocks)

    system_prompt = f"""
Tu esi MI asistents, kas atbild par Biofruitnet PDF kolekciju.

Atbildi tikai balstoties uz sniegto kontekstu.

Ja atbilde nav kontekstā:
"Diemžēl šāda informācija manos PDF nav atrasta."

Atbildi vienmēr latviešu valodā.

Konteksts:
{context_text}
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history[-6:])
    messages.append({"role": "user", "content": question})

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0
    )

    answer = resp.choices[0].message.content.strip()

    return {
        "answer": answer,
        "sources": sources[:10]
    }
