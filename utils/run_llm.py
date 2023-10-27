def run_llm(llm, question, context):
    output = llm(f""""Verwende die folgenden Informationen, um die Frage des Benutzers zu beantworten.
Wenn du die Antwort nicht weißt, sag einfach, dass du es nicht weißt, versuche nicht, eine Antwort zu erfinden.

Kontext: {context}
Frage: {question}

Gib nur die hilfreichen Informationen aus dem Kontext wortwörtlich wieder und nichts anderes.
Hilfreiche Antwort: """, temperature= 0.01, top_k = 10000, repeat_penalty = 1.1, max_tokens = 200, echo=False)
    return output