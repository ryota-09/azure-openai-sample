from fastapi import FastAPI
import os
import openai
import time
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    ComplexField,
    CorsOptions,
    SearchIndex,
    ScoringProfile,
    SearchFieldDataType,
    SearchField,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswVectorSearchAlgorithmConfiguration
)
from azure.search.documents import SearchClient

app = FastAPI()

openai.api_type = "azure"
openai.api_base = "https://sample-instance.openai.azure.com/"
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")

message_text = [
    {"role":"system","content":"##役割 \n あなたは優秀なコピーライターです。ユーザーが提供するお題に対してのコピーを考えてください。\n ## 制約条件 \n ・コピーは20文字以内です。\n・漢字の割合を30%、ひらがなの割合を70%にします。\n・魅力的かつ印象的なコピーを考えてください。"},
    { "role": "user", "content": "ゆず味のアイスキャンディー" },
    { "role": "assistant", "content": "爽やかなゆずの口どけ" },
    {  "role": "user", "content": "サバ味のソフトクリーム"}
]

@app.get("/hello")
async def hello():
    response = openai.ChatCompletion.create(
        engine="sampleChatModel1",
        messages = message_text,
        temperature=1,
        max_tokens=800,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    result = response["choices"][0]["message"]["content"]
    return {"message": result}

endpoint = os.getenv("AZURE_SEARCH_SERVICE_ENDPOINT")
key = os.getenv("AZURE_SEARCH_API_KEY")
client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(key))

DOCUMENT = [
    {
        "id": "1",
        "title": "泣く女",
        "artist": "パブロ・ピカソ",
        "description": "ピカソの代表作の一つ。ピカソが愛した女性マリー・テレーズ・ヴァルテの肖像画。ピカソが最も愛した女性であり、彼女の死後、ピカソは自らの死までこの作品を手元に置いていた。",
        "category": "絵画",
        "creation_date": "1937年"
    },
    {
        "id": "2",
        "title": "ゲルニカ",
        "artist": "パブロ・ピカソ",
        "description": "スペイン内戦中の1937年に起きたバスク地方ゲルニカの爆撃をテーマにした作品。ピカソはスペイン政府からこの作品をスペイン政府館に展示するよう依頼されたが、ピカソはこれを拒否し、スペイン政府館にはこの作品が展示されることはなかった。",
        "category": "絵画",
        "creation_date": "1937年"
    },
    {
        "id": "3",
        "title": "老いたギタリスト",
        "artist": "パブロ・ピカソ",
        "description": "ピカソの代表作の一つ。ピカソが晩年に描いた作品であり、ピカソが晩年に描いた作品の中でも最も有名な作品の一つである。",
        "category": "絵画",
        "creation_date": "1903-1904年"
    }
]

name = "artworks"

fields = [
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="title", type=SearchFieldDataType.String),
        SearchableField(name="artist", type=SearchFieldDataType.String),
        SearchableField(name="description", type=SearchFieldDataType.String),
        SearchableField(name="category", type=SearchFieldDataType.String),
        SearchableField(name="creation_date", type=SearchFieldDataType.String),
    ]
cors_options = CorsOptions(allowed_origins=["*"], max_age_in_seconds=120)
scoring_profiles = []

index = SearchIndex(
    name=name,
    fields=fields,
    scoring_profiles=scoring_profiles,
    cors_options=cors_options,
)

@app.get("/search")
async def search():
    result = client.create_index(index)
    return {"index": result}

@app.get("/upload")
async def upload():
    search_client = SearchClient(endpoint,name, credential=AzureKeyCredential(key))
    result = search_client.upload_documents(documents=DOCUMENT)
    return {"upload": result[0]}

@app.get("/searchByKeyword")
async def searchByKeyword():
    client = SearchClient(endpoint=endpoint, index_name=name, credential=AzureKeyCredential(key))
    results = client.search(search_text="泣く女", include_total_count=True)
    print("トータル件数:",results.get_count())
    for result in results:
        print("id:",result["id"])
        print("title:",result["title"])
        print("artist:",result["artist"])
        print("description:",result["description"])
        print("category:",result["category"])
        print("creation_date:",result["creation_date"])
    return {"search": results}

index_name = "artworks-vector"

# ベクトル検索用の設定
vector_search = VectorSearch(
    algorithm_configurations=[
        HnswVectorSearchAlgorithmConfiguration(name="vectorConfig", kind="hnsw", parameters={"m": 4, "efConstruction": 400, "efSearch": 500, "metric": "cosine"})
    ]
)

vector_fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchableField(name="title", type=SearchFieldDataType.String, searchable=True),
    SearchField(name="title_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_configuration="vectorConfig"),
    SearchableField(name="artist", type=SearchFieldDataType.String, searchable=True),
    SearchableField(name="description", type=SearchFieldDataType.String, searchable=True),
    SearchField(name="description_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_configuration="vectorConfig"),
    SearchableField(name="category", type=SearchFieldDataType.String, searchable=True),
    SimpleField(name="creation_date", type=SearchFieldDataType.String),
]

@app.get("/createIndexWithVector")
async def createIndexWithVector():
    vector_index = SearchIndex(name=index_name, fields=vector_fields, scoring_profiles=scoring_profiles, vector_search=vector_search, cors_options=cors_options)
    result = client.create_index(vector_index)
    return {"index": "created"}

VECOR_DOCUMENT = [
    {
        "id": "1",
        "title": "泣く女",
        "title_vector": "",
        "artist": "パブロ・ピカソ",
        "description": "ピカソの代表作の一つ。ピカソが愛した女性マリー・テレーズ・ヴァルテの肖像画。ピカソが最も愛した女性であり、彼女の死後、ピカソは自らの死までこの作品を手元に置いていた。",
        "description_vector": "",
        "category": "絵画",
        "creation_date": "1937年"
    },
    {
        "id": "2",
        "title": "ゲルニカ",
        "title_vector": "",
        "artist": "パブロ・ピカソ",
        "description": "スペイン内戦中の1937年に起きたバスク地方ゲルニカの爆撃をテーマにした作品。ピカソはスペイン政府からこの作品をスペイン政府館に展示するよう依頼されたが、ピカソはこれを拒否し、スペイン政府館にはこの作品が展示されることはなかった。",
        "description_vector": "",
        "category": "絵画",
        "creation_date": "1937年"
    },
    {
        "id": "3",
        "title": "老いたギタリスト",
        "title_vector": "",
        "artist": "パブロ・ピカソ",
        "description": "ピカソの代表作の一つ。ピカソが晩年に描いた作品であり、ピカソが晩年に描いた作品の中でも最も有名な作品の一つである。",
        "description_vector": "",
        "category": "絵画",
        "creation_date": "1903-1904年"
    }
]

@app.get("/uploadVector")
async def uploadVector():
    for doc in VECOR_DOCUMENT:
        # ベクトル化
        response_title = openai.Embedding.create(input=[doc["title"]], engine="sampleEmbedding")
        doc["title_vector"] = response_title["data"][0]["embedding"]
        time.sleep(10)
        response_description = openai.Embedding.create(input=[doc["description"]], engine="sampleEmbedding")
        doc["description_vector"] = response_description["data"][0]["embedding"]
        time.sleep(10)
    client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(key))
    result = client.upload_documents(documents=VECOR_DOCUMENT)
    return {"upload": result[0]}
