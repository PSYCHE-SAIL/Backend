from google.cloud import aiplatform
from langchain_google_vertexai import VertexAIEmbeddings
from langchain_google_vertexai import (
    VectorSearchVectorStore,
)
# TODO : Set values as per your requirements
# Project and Storage Constants
PROJECT_ID = "psychesail-421404"
REGION = "asia-south1"
BUCKET = "storechat-psychesail"
BUCKET_URI = f"gs://{BUCKET}"

# The number of dimensions for the textembedding-gecko@003 is 768
# If other embedder is used, the dimensions would probably need to change.
DIMENSIONS = 768
MINIMUM_SIMILARITY = 0.8 
# Index Constants
DISPLAY_NAME = "chatStoring"
DEPLOYED_INDEX_ID = "chatstore_1714945519155"


aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

embedding_model = VertexAIEmbeddings(model_name="textembedding-gecko@003")


# NOTE : This operation can take upto 30 seconds
my_index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name=DISPLAY_NAME,
    dimensions=DIMENSIONS,
    approximate_neighbors_count=1,
    distance_measure_type="DOT_PRODUCT_DISTANCE",
    index_update_method="STREAM_UPDATE",  # allowed values BATCH_UPDATE , STREAM_UPDATE
)

# Create an endpoint
my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name=f"{DISPLAY_NAME}-endpoint", public_endpoint_enabled=True
)

# NOTE : This operation can take upto 20 minutes
my_index_endpoint = my_index_endpoint.deploy_index(
    index=my_index, deployed_index_id=DEPLOYED_INDEX_ID
)

# TODO : replace 1234567890123456789 with your acutial index ID
my_index = aiplatform.MatchingEngineIndex("5895071174739623936")

# TODO : replace 1234567890123456789 with your acutial endpoint ID
my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint("5922374247480557568")


def search_or_add_messages(messages_string):

  # Connect to vector store
  vector_store = VectorSearchVectorStore.from_components(
      project_id=PROJECT_ID,
      region=REGION,
      gcs_bucket_name=BUCKET,
      index_id=DEPLOYED_INDEX_ID,
      embedding=embedding_model,
  )

  # Search for similar messages in the concatenated string
  similar_messages = vector_store.similarity_search(messages_string)

  results = {}
  for message in messages_string.splitlines():  # Split by newline for individual messages
    # Check if a similar message exists with high enough similarity
    if similar_messages and max(similar_messages.values()) >= MINIMUM_SIMILARITY:
      most_similar_id = max(similar_messages, key=similar_messages.get)
      results[message] = most_similar_id
    else:
      # Add the message with a new ID (implementation detail left out for brevity)
      new_message = vector_store.add_texts(texts=messages_string, is_complete_overwrite=True)
  # Replace with your add_message function
      results[message] = new_message

  return results


# Example usage
def getChatID(messages):

    messages_string = "\n".join(messages)  # Join messages with newline delimiter
    results = search_or_add_messages(messages_string)
    print(results)
    return(results)