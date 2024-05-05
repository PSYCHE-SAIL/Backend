from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from typing import List

def create_index(project_id: str, location: str, display_name: str, dimensions: int) -> aiplatform.MatchingEngineIndex:

  aiplatform.init(project=project_id, location=location)
  return aiplatform.MatchingEngineIndex.create_tree_ah_index(
      display_name=display_name,
      contents_delta_uri=None,  
      dimensions=dimensions,
      nearest_neighbors_count=1  
  )

def embed_messages(messages: List[str], model_name: str, task_type: str, title: str = "", output_dimensionality=None) -> List[List[float]]:

  model = TextEmbeddingModel.from_pretrained(model_name)
  embeddings = []
  for message in messages:
    text_embedding_input = TextEmbeddingInput(
        task_type=task_type, title=title, text=message
    )
    kwargs = (
        dict(output_dimensionality=output_dimensionality)
        if output_dimensionality
        else {}
    )
    embedding = model.get_embeddings([text_embedding_input], **kwargs)[0].values
    embeddings.append(embedding)
  return embeddings

def search_and_add(index: aiplatform.MatchingEngineIndex, message: str, message_id: str, threshold: float) -> str:

  embedding = embed_messages([message], model_name=MODEL, task_type=TASK,  output_dimensionality=OUTPUT_DIMENSIONALITY)[0]
  results = index.match(
    deployed_index_id=index,
    queries=embedding,
    num_neighbors=1,
    )

  if results.distances[0] <= 1 - threshold:
    return results.ids[0]  # Return matched document ID
  else:

    index.add_embeddings(embeddings=[embedding], ids=[message_id])
    tree_ah_index = index.update_embeddings(
    contents_delta_uri=EMBEDDINGS_UPDATE_URI,
)
    return message_id  # Return provided message ID (added)

project_id = "psychesail-421404"
location = "us-central1"
index_display_name = "chatStoring"
MODEL = "text-embedding-preview-0409"  
TASK = "SEMANTIC_SIMILARITY" 
OUTPUT_DIMENSIONALITY = 768 
DEPLOYED_INDEX_ID = 3216512 
message_threshold = 0.8

index = create_index(project_id, location, index_display_name, OUTPUT_DIMENSIONALITY)

my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name=index_display_name,
    public_endpoint_enabled=False,
)

my_index_endpoint = index.deploy_index(
    index=index, deployed_index_id=DEPLOYED_INDEX_ID
)

def getChatID(messages):

    messages = ["This is a sample message 1", "This is a similar message 2", "This is a new message 3"]
    for message, message_id in zip(messages, ["message_1", "message_2", "message_3"]):
        matched_id = search_and_add(index, message, message_id, message_threshold)
        print(f"Message '{message}' {'matched' if matched_id != message_id else 'added'} with ID: {matched_id}")
    return matched_id
