from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def load_retriever():

    loader = PyPDFLoader(file_path="Medical_book.pdf")

    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=550)

    text_chunks = splitter.split_documents(documents=docs)

    embeddings_model = HuggingFaceEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=text_chunks,
        embedding=embeddings_model,
        persist_directory="./chromaDB",
    )

    return vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 5})


def load_model():
    return ChatGroq(model_name="llama-3.3-70B-versatile")


def format_documents(docs):
    return "\n".join(doc.page_content for doc in docs)


def format_history(chat_history):
    if not chat_history:
        return "No conversation yet"
    else:
        return "\n".join(
            f"User : {turn['user']}\nAssistant : {turn['assistant']}"
            for turn in chat_history
        )


def build_chain(retriever, model, chat_history: list[dict]):

    prompt = PromptTemplate(
        template="""
        Answer the {question} in detail
        The answer should be in points well structured
        It should include all the precautions, side-effects, advantage.
        It should clearly mention what this medicine does and why is it used for 
        Tell the dosage the medicine should be taken in and how should you take it
        Maintain the conversation history to keep the conversation meaningful

        If the question is not related to medicines then answer then question in short points
        You dont need to print chat history in the answer

        chat_history : {chat_history}

        Context:{context}
        Answer:
        """,
        input_variables=["question", "chat_history", "context"],
    )

    retrievel_chain = RunnableParallel(
        {"context": retriever | format_documents, "question": RunnablePassthrough()}
    )

    def add_history(inputs):
        return {
            "chat_history": format_history(chat_history),
            "context": inputs["context"],
            "question": inputs["question"],
        }

    return retrievel_chain | add_history | prompt | model | StrOutputParser()


def get_response(question: str, retriever, model, chat_history: list[dict]):
    chain = build_chain(retriever=retriever, model=model, chat_history=chat_history)
    return chain.invoke(question)


retriver = load_retriever()
model = load_model()
chat_history = []

while True:
    question = input("You :")
    if question.lower() == "exit":
        break

    answer = get_response(question, retriver, model, chat_history)

    print(f"Assistant : {answer}\n")
    chat_history.append({"user": question, "assistant": answer})
