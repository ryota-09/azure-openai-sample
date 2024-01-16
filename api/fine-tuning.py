import json
import tiktoken
import numpy as np
from collections import defaultdict
import openai
import os

openai.api_type = "azure"
openai.api_base = "https://sample-instance.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")

# トレーニングセットの準備
with open('train.json', 'r', encoding="utf-8") as f:
    training_dataset = [json.loads(line) for line in f]
    
# バリデーションセットの準備
with open('valid.json', 'r', encoding="utf-8") as f:
    validation_dataset = [json.loads(line) for line in f]

encoding = tiktoken.get_encoding("cl100k_base")

# トークン数の計算
def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens

def num_assistant_tokens_from_messages(messages):
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens

files = ["train.json", "valid.json"]

for file in files:
  with open(file, 'r', encoding="utf-8") as f:
    dataset = [json.loads(line) for line in f]
  total_tokens = []
  assistant_tokens = []
  
  for ex in dataset:
    messages = ex.get("messages", {})
    total_tokens.append(num_tokens_from_messages(messages))
    assistant_tokens.append(num_assistant_tokens_from_messages(messages))

training_file_id = "file-xxxxx"
validation_file_id = "file-xxxxx"

response = openai.FineTuningJob.create(
  training_file=training_file_id,
  validation_file=validation_file_id,
  model="gpt-35-turbo-0613"
)

job_id = response["id"]

