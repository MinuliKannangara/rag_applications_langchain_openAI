from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Initialize the ChatOpenAI model with the API key
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    api_key=""
)

# Initialize the embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# create and embed documents
# Define a list of documents with content and metadata
documents = [
    Document(
        page_content="The T20 World Cup 2024 is in full swing, bringing excitement and drama to cricket fans worldwide.India's team, captained by Rohit Sharma, is preparing for a crucial match against Ireland, with standout player Jasprit Bumrah expected to play a pivotal role in their campaign.The tournament has already seen controversy, particularly concerning the pitch conditions at Nassau County International Cricket Stadium in New York, which came under fire after a low-scoring game between Sri Lanka and South Africa.",
        metadata={"source": "cricket news"},
    ),
    Document(
        page_content="The world of football is buzzing with excitement as major tournaments and league matches continue to captivate fans globally.In the UEFA Champions League, the semi-final matchups have been set, with defending champions Real Madrid set to face Manchester City, while Bayern Munich will take on Paris Saint-Germain.Both ties promise thrilling encounters, featuring some of the best talents in world football.",
        metadata={"source": "football news"},
    ),
    Document(
        page_content="As election season heats up, the latest developments reveal a highly competitive atmosphere across several key races.The presidential election has seen intense campaigning from all major candidates, with recent polls indicating a tight race.Incumbent President Jane Doe is seeking re-election on a platform of economic stability and healthcare reform, while her main rival, Senator John Smith, focuses on education and climate change initiatives.",
        metadata={"source": "election news"},
    ),
    Document(
        page_content="The AI revolution continues to transform industries and reshape the global economy.Significant advancements in artificial intelligence have led to breakthroughs in healthcare, with AI-driven diagnostics improving patient outcomes and reducing costs.Autonomous systems are becoming increasingly prevalent in logistics and transportation, enhancing efficiency and safety.",
        metadata={"source": "ai revolution news"},
    ),
]

# Create a vector store using the documents and embedding model
vectorstore = Chroma.from_documents(
    documents,
    embedding=embedding_model,
)

# perform similarty search by giving the text
results = vectorstore.similarity_search("test match")

for result in results:
  print("------------------------")
  print(result.page_content)
  print(result.metadata)

# perform the similarity search by giving the vector (embedding)
# Embed a query using the embedding model
query_embedding = embedding_model.embed_query("machine learning")

results = vectorstore.similarity_search_by_vector(query_embedding)

for result in results:
    print("------------------------")
    print(result.page_content)
    print(result.metadata)

# Create a retriever from the vector
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

# Perform batch retrieval using the retriever
batch_results = retriever.batch(["machine learning", "test match"])

for result in batch_results:
  print("------------------------")
  for doc in result:
    print(doc.page_content)
    print(doc.metadata)

# create prompt template
# Define a message template for the chatbot
message = """
Answer this question using the provided context only.

{question}

Context:
{context}
"""

# Create a chat prompt template from the message
prompt = ChatPromptTemplate.from_messages([("human", message)])

# Chain Retriever and Prompt Template with LLM
chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | llm

response = chain.invoke("current state of 2024 t20 world cup")

print(response.content)