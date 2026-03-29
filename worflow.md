User question: "What is aspirin?"
        │
        ▼
 retriever.invoke("What is aspirin?")
   → searches ChromaDB vector store
   → returns 5 most relevant PDF chunks (k=5, MMR search)

        │
        ▼
 format_documents(docs)
   → joins all 5 chunks into one big string
   → this becomes {context}

        │
        ▼
 add_history(inputs)
   → reads chat_history list
   → formats it as "User: ...\nAssistant: ..."
   → now we have {question}, {context}, {chat_history}

        │
        ▼
 PromptTemplate fills in the template
   → produces a full prompt string

        │
        ▼
 ChatGroq (llama-3.3-70b) generates the answer

        │
        ▼
 StrOutputParser() → returns plain string