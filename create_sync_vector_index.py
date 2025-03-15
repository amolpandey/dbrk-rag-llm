# Databricks notebook source
!pip install databricks-vectorsearch --quiet
!pip install mlflow[databricks] --quiet
!pip install databricks-agents --quiet
!pip install databricks-langchain --quiet
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Read the workspace notebook for building Vertor Store
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace
import base64

def get_notebook_code_base(notebook_path):
    myToken = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
    databricksURL = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiUrl().getOrElse(None)
    w = WorkspaceClient(host=databricksURL, token=myToken)
    export_response = w.workspace.export(notebook_path, format=workspace.ExportFormat.SOURCE)
    code_data = base64.b64decode(export_response.content).decode('utf-8')
    return code_data
    
def get_text_from_folder_files(path: str) -> dict:
    iCount = 0
    txt_data = []
    for f  in dbutils.fs.ls(path):
        data_dict = {}
        file_name = f.name.replace('/','')
        if file_name not in ['distance_to_coast','failure_mode_effect_analysis','interpolated_bushfire_consequence','service_history']:
            code_base = get_notebook_code_base(f'{f.path}tables/{file_name}'.replace('file:',''))
            data_dict['file_no'] = iCount
            data_dict['file_name'] = file_name
            data_dict['file_path'] = f'{f.path}tables/{file_name}'.replace('file:','')
            data_dict['file_text'] = code_base
            txt_data.append(data_dict)
            iCount += 1

    return txt_data

src_data_path = 'file:/Workspace/Users/username@company.com.au/project/notebooks/demo/'
codebase_result = get_text_from_folder_files(src_data_path)

# COMMAND ----------

# DBTITLE 1,Chunk the data into fixed size window
# Refrence: https://jillanisofttech.medium.com/optimizing-chunking-strategies-for-retrieval-augmented-generation-rag-applications-with-python-c3ab5060d3e4

def fixed_size_chunking(text, max_tokens):
    words = text.split()
    chunks = []
    current_chunk = []
    current_tokens = 0

    for word in words:
        current_tokens += len(word.split())
        if current_tokens <= max_tokens:
            current_chunk.append(word)
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_tokens = len(word.split())
    # Append the last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    return chunks

chunk_data = []
max_tokens = 100
iCount = 0
for dt in codebase_result:
    text = dt['file_text']
    chunks = fixed_size_chunking(text, max_tokens)
    for i, chunk in enumerate(chunks):
        data_dict = {}
        data_dict['row_no'] = iCount
        data_dict['file_no'] = dt['file_no']
        data_dict['file_name'] = dt['file_name']
        data_dict['file_path'] = dt['file_path']
        data_dict['chunk_no'] = i + 1
        data_dict['chunk_text'] = chunk
        chunk_data.append(data_dict)
        iCount += 1

for c in chunk_data:
    if c['file_no'] == 0:
        print(c)

# COMMAND ----------

# DBTITLE 1,Prepare the delta table for data store
data_sdf = spark.createDataFrame(chunk_data)
data_sdf = data_sdf.selectExpr('row_no','file_no','file_name','file_path', 'chunk_no','chunk_text')

data_sdf.write \
.mode("overwrite") \
.option("overwriteSchema", "true") \
.saveAsTable(f'catalog.schema.eval_rag_data_codebase')

# COMMAND ----------

# DBTITLE 1,Enable CDC on the new created table
# Note: CDC has to be enabled on the underlying table for the building of vector search index
spark.sql('ALTER TABLE catalog.schema.eval_rag_data_codebase SET TBLPROPERTIES (delta.enableChangeDataFeed = true)')

# COMMAND ----------

# DBTITLE 1,Create a Vector Search EndPoint
# Refrence: https://docs.databricks.com/aws/en/generative-ai/create-query-vector-search#create-a-vector-search-endpoint
from databricks.vector_search.client import VectorSearchClient

VECTOR_SEARCH_ENDPOINT_NAME = 'eval-codebase-ep'
vsc = VectorSearchClient(disable_notice=True)
vsc.create_endpoint(name=VECTOR_SEARCH_ENDPOINT_NAME, endpoint_type="STANDARD")
# Please note the "Provisoning" can take a while so be patient. Approx 10 mins

# COMMAND ----------

# DBTITLE 1,Create a vector index from the delta table in the new end point
client = VectorSearchClient(disable_notice=True)

index = client.create_delta_sync_index(
  endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME,
  source_table_name="catalog.schema.eval_rag_data_codebase",
  index_name="catalog.schema.eval_rag_data_codebase_vector_idx",
  pipeline_type="TRIGGERED",
  primary_key="row_no",
  embedding_source_column="chunk_text",
  embedding_model_endpoint_name="databricks-bge-large-en",
  columns_to_sync=['row_no', 'file_no', 'file_name', 'file_path', 'chunk_no', 'chunk_text'] # to sync only the primary key and the embedding column
)

# COMMAND ----------

# DBTITLE 1,Evaluate: Test the new vector index for its similarity score
client = VectorSearchClient(disable_notice=True)
index = client.get_index(endpoint_name=VECTOR_SEARCH_ENDPOINT_NAME, index_name='catalog.schema.eval_rag_data_codebase_vector_idx')
print('Vector Index Name: ' + index.name)

result = index.similarity_search(
    query_text='Where is node_attributes_wcs used ?',
    columns=['row_no', 'file_name' , 'chunk_no',  'chunk_text'],
    num_results=3
)
for d in result['result']['data_array']:
    print(d[0], d[1], d[2], d[3][:50])


# COMMAND ----------

# DBTITLE 1,Config Prepration for the LLM
# For this first basic demo, we'll keep the configuration as a minimum. In real app, you can make all your RAG as a param (such as your prompt template to easily test different prompts!)
chain_config = {
    "llm_model_serving_endpoint_name": "databricks-meta-llama-3-1-70b-instruct",  # the foundation model we want to use
    "vector_search_endpoint_name": VECTOR_SEARCH_ENDPOINT_NAME,  # the endoint we want to use for vector search
    "vector_search_index": 'catalog.schema.eval_rag_data_codebase_vector_idx',
    "llm_prompt_template": """You are an assistant that answers questions. Use the following pieces of retrieved context to answer the question. Some pieces of context may be irrelevant, in which case you should not use them to form the answer.\n\nContext: {context}""",
}

# COMMAND ----------

# DBTITLE 1,Build the Langchain retriever
from databricks.vector_search.client import VectorSearchClient
from databricks_langchain.vectorstores import DatabricksVectorSearch
from langchain.schema.runnable import RunnableLambda
from langchain_core.output_parsers import StrOutputParser
import mlflow

## Enable MLflow Tracing
mlflow.langchain.autolog()

## Load the chain's configuration
model_config = mlflow.models.ModelConfig(development_config=chain_config)

## Turn the Vector Search index into a LangChain retriever
vector_search_as_retriever = DatabricksVectorSearch(
    endpoint=model_config.get("vector_search_endpoint_name"),
    index_name=model_config.get("vector_search_index"),
    columns=['row_no', 'file_name' , 'chunk_no',  'chunk_text'],
).as_retriever(search_kwargs={"k": 3})

# Method to format the docs returned by the retriever into the prompt (keep only the text from chunks)
def format_context(docs):
    chunk_contents = [f"Passage: {d.page_content}\n" for d in docs]
    return "".join(chunk_contents)

#Let's try our retriever chain:
relevant_docs = (vector_search_as_retriever | RunnableLambda(format_context)| StrOutputParser()).invoke('Where is node_attributes_wcs used ?')

print(relevant_docs)

# COMMAND ----------

# DBTITLE 1,Preparing the foundational LLM
from langchain_core.prompts import ChatPromptTemplate
from databricks_langchain.chat_models import ChatDatabricks
from operator import itemgetter

prompt_template = """You are an assistant that answers questions. Use the following pieces of retrieved context to answer the question. Some pieces of context may be irrelevant, in which case you should not use them to form the answer.\n\nContext: {context}"""
prompt = ChatPromptTemplate.from_messages(
    [  
        ("system", prompt_template), # Contains the instructions from the configuration
        ("user", "{question}") #user's questions
    ]
)

# Our foundation model answering the final prompt
model = ChatDatabricks(
    endpoint='databricks-meta-llama-3-1-70b-instruct',
    extra_params={"temperature": 0.01, "max_tokens": 500}
)

#Let's try our prompt:
answer = (prompt | model | StrOutputParser()).invoke({'question':'Where is node_attributes_wcs used ?', 'context': ''})
print(answer)

# COMMAND ----------

# DBTITLE 1,Prepare the RAG Chat Model
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

# COMMAND ----------

# DBTITLE 1,Evaluate the RAG Model for Responses based on the Codebase
# Let's give it a try:
input_example = {"messages": [ {"role": "user", "content": "What source table names build up the vegetation_area table ?"}]}
answer = chain.invoke(input_example)
print(answer)

# COMMAND ----------

# DBTITLE 1,Register the model to Unity Catalog Models
from mlflow.models.resources import DatabricksVectorSearchIndex, DatabricksServingEndpoint
import mlflow
import os

mlflow.set_registry_uri('databricks-uc') # Note: This is required to publish mode in the unity catalog vs Workspace catalog.

# Log the model to MLflow
with mlflow.start_run(run_name="rag_bot_codebase"):
  logged_chain_info = mlflow.langchain.log_model(
          #Note: In classical ML, MLflow works by serializing the model object.  In generative AI, chains often include Python packages that do not serialize.  Here, we use MLflow's new code-based logging, where we saved our chain under the chain notebook and will use this code instead of trying to serialize the object.
          lc_model=os.path.join(os.getcwd(), 'chain.py'),  # Chain code file e.g., /path/to/the/chain.py 
          model_config=chain_config, # Chain configuration 
          artifact_path="chain", # Required by MLflow, the chain's code/config are saved in this directory
          input_example=input_example,
          example_no_conversion=True,  # Required by MLflow to use the input_example as the chain's schema,
          # Specify resources for automatic authentication passthrough
          resources=[
            DatabricksVectorSearchIndex(index_name='catalog.schema.eval_rag_data_codebase_vector_idx'),
            DatabricksServingEndpoint(endpoint_name='databricks-meta-llama-3-1-70b-instruct')
          ]
      )

MODEL_NAME = "rag_bot_codebase"
MODEL_NAME_FQN = f"catalog.schema.{MODEL_NAME}"
# Register to UC
uc_registered_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=MODEL_NAME_FQN)

# COMMAND ----------

# DBTITLE 1,Deploy the model to Serving endpoint for publishing and evaluating
from databricks import agents
# Deploy to enable the Review APP and create an API endpoint
# Note: scaling down to zero will provide unexpected behavior for the chat app. Set it to false for a prod-ready application.
deployment_info = agents.deploy(MODEL_NAME_FQN, model_version=uc_registered_model_info.version, scale_to_zero=True)

instructions_to_reviewer = f"""## Instructions for Testing the Custom Evaluation Assistant chatbot

Your inputs are invaluable for the development team. By providing detailed feedback and corrections, you help us fix issues and improve the overall quality of the application. We rely on your expertise to identify any gaps or areas needing enhancement."""

# Add the user-facing instructions to the Review App
agents.set_review_instructions(MODEL_NAME_FQN, instructions_to_reviewer)