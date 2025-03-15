from databricks.vector_search.client import VectorSearchClient
from databricks_langchain.vectorstores import DatabricksVectorSearch
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import mlflow

## Enable MLflow Tracing
mlflow.langchain.autolog()

model_config = mlflow.models.ModelConfig(development_config="rag_chain_config.yaml")


# Turn the Vector Search index into a LangChain retriever
vector_search_as_retriever = DatabricksVectorSearch(
    endpoint=model_config.get("vector_search_endpoint_name"),
    index_name=model_config.get("vector_search_index"),
    columns=['row_no', 'file_name', 'file_path' , 'chunk_no',  'chunk_text'],
).as_retriever(search_kwargs={"k": 3}) # Number of search results that the retriever returns
# Enable the RAG Studio Review App and MLFlow to properly display track and display retrieved chunks for evaluation
mlflow.models.set_retriever_schema(primary_key="row_no", text_column="chunk_text", doc_uri="file_path")

# Method to format the docs returned by the retriever into the prompt (keep only the text from chunks)
def format_context(docs):
    chunk_contents = [f"Passage: {d.page_content}\n" for d in docs]
    return "".join(chunk_contents)

from langchain_core.prompts import ChatPromptTemplate
from databricks_langchain.chat_models import ChatDatabricks
from operator import itemgetter

prompt = ChatPromptTemplate.from_messages(
    [
        (  # System prompt contains the instructions
            "system",
            """You are an assistant that answers questions. Use the following pieces of retrieved context to answer the question. Some pieces of context may be irrelevant, in which case you should not use them to form the answer.

Context: {context}""",
        ),
        # User's question
        ("user", "{question}"),
    ]
)

# Our foundation model answering the final prompt
model = ChatDatabricks(
    endpoint=model_config.get("llm_model_serving_endpoint_name"),
    extra_params={"temperature": 0.01, "max_tokens": 500}
)

# Return the string contents of the most recent messages: [{...}] from the user to be used as input question
def extract_user_query_string(chat_messages_array):
    return chat_messages_array[-1]["content"]

# RAG Chain
chain = (
    {
        "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
        "context": itemgetter("messages")
        | RunnableLambda(extract_user_query_string)
        | vector_search_as_retriever
        | RunnableLambda(format_context),
    }
    | prompt
    | model
    | StrOutputParser()
)

# Tell MLflow logging where to find your chain.
mlflow.models.set_model(model=chain)