from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# Initialize the language model
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# Define a prompt template
system_prompt = (
    "Use the following pieces of retrieved context to answer "
    "the user's question. If you don't know the answer, say that you "
    "don't know. Keep the answer concise.\n\n{context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Create the chains
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Query the RAG pipeline
query = "What information is available about [specific topic]?"
response = rag_chain.invoke({"input": query})
print(response["answer"])