from fastapi import FastAPI
import os
import openai
import time
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    CorsOptions,
    SearchIndex,
    SearchFieldDataType,
    SearchField,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswVectorSearchAlgorithmConfiguration,
    SemanticConfiguration,
    SemanticField,
    SemanticSettings,
    PrioritizedFields
)
from azure.search.documents import SearchClient
from azure.search.documents.models import Vector

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.azuresearch import AzureSearch
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

from langchain.chat_models import AzureChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import (
    SystemMessage,
    HumanMessage
)

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

key_word = "楽器を含む絵画"

@app.get("/searchWithVector")
async def searchWithVector():
    response_search_word_vector = openai.Embedding.create(input=key_word, engine="sampleEmbedding")
    client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(key))
    results = client.search(search_text="", include_total_count=True, vectors=[Vector(
        value=response_search_word_vector["data"][0]["embedding"],
        k=3,
        fields="description_vector"
    )])
    print("件数:",results.get_count())
    for result in results:
        print("id:",result["id"])
        print("title:",result["title"])
        print("artist:",result["artist"])
        print("description:",result["description"])
        print("category:",result["category"])
        print("creation_date:",result["creation_date"])

semantic_index_name = "index-semantic"

semantic_fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),
    SearchableField(name="title", type=SearchFieldDataType.String, searchable=True, retrievable=True, analyzer_name="ja.microsoft"),
    SearchField(name="title_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_configuration="vectorConfig"),
    SearchableField(name="artist", type=SearchFieldDataType.String, searchable=True, retrievable=True, analyzer_name="ja.microsoft"),
    SearchableField(name="description", type=SearchFieldDataType.String, searchable=True, retrievable=True, analyzer_name="ja.microsoft"),
    SearchField(name="description_vector", type=SearchFieldDataType.Collection(SearchFieldDataType.Single), searchable=True, vector_search_dimensions=1536, vector_search_configuration="vectorConfig")
]

semantic_config = SemanticConfiguration(
    name="my-semantic-config",
    prioritized_fields=PrioritizedFields(title_field=SemanticField(field_name="title"), prioritized_content_fields=[SemanticField(field_name="description")]),
)

semantic_settings = SemanticSettings(configurations=[semantic_config])

@app.get("/createIndexWithSemantic")
async def createIndexWithSemantic():
    client = SearchIndexClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    semantic_index = SearchIndex(name=semantic_index_name, fields=semantic_fields,semantic_settings=semantic_settings,scoring_profiles=scoring_profiles, vector_search=vector_search, cors_options=cors_options)
    result = client.create_index(semantic_index)
    return {"index": "created", "result": result}


SEMANTIC_DOCUMENT = [
    {
        "id": "1",
        "title": "フィンセント・ファン・ゴッホ",
        "title_vector": "",
        "description": "フィンセント・ウィレム・ファン・ゴッホは、オランダの画家。後期印象派の画家として知られる。代表作に『ひまわり』『星月夜』『刈った麦の束』などがある。",
        "description_vector": "",
    },
    {
        "id": "2",
        "title": "パブロ・ピカソ",
        "title_vector": "",
        "artist": "パブロ・ピカソ",
        "description": "パブロ・ピカソは、スペインの画家、彫刻家、陶芸家、舞台美術家、詩人。キュビスムの創始者として知られる。代表作に『ゲルニカ』『泣く女』『老いたギタリスト』などがある。",
        "description_vector": "",
    },
    {
        "id": "3",
        "title": "ジャン＝フランソワ・ミレー",
        "title_vector": "",
        "artist": "パブロ・ピカソ",
        "description": "ミレーは、フランスの画家。バルビゾン派の画家として知られる。代表作に『落穂拾い』『晩鐘』『種まく人』などがある。",
        "description_vector": ""
    }
]

@app.get("/uploadSemantic")
async def uploadSemantic():
    for doc in SEMANTIC_DOCUMENT:
        # ベクトル化
        response_title = openai.Embedding.create(input=[doc["title"]], engine="sampleEmbedding")
        doc["title_vector"] = response_title["data"][0]["embedding"]
        time.sleep(30)
        response_description = openai.Embedding.create(input=[doc["description"]], engine="sampleEmbedding")
        doc["description_vector"] = response_description["data"][0]["embedding"]
        time.sleep(30)
    client = SearchClient(endpoint=endpoint, index_name=semantic_index_name, credential=AzureKeyCredential(key))
    result = client.upload_documents(documents=SEMANTIC_DOCUMENT)
    return {"upload": result[0]}

semantic_search_word = "戦争をテーマにした絵画"

@app.get("/searchWithSemantic")
async def searchWithSemantic():
    client = SearchClient(endpoint=endpoint, index_name=semantic_index_name, credential=AzureKeyCredential(key))
    results = client.search(search_text=semantic_search_word, select=["title", "description"],query_type="semantic", query_language="ja-JP", semantic_configuration_name="my-semantic-config", query_caption="extractive", query_answer="extractive", top=10)
    for result in results:
        print("title:",result["title"])
        print("description:",result["description"])
        captions = result["@search.captions"]
        if captions:
            print("caption:",captions[0])

@app.get("/searchWithSemanticAndVector")
async def searchWithSemanticAndVector():
    response_search_word_vector = openai.Embedding.create(input=semantic_search_word, engine="sampleEmbedding")
    client = SearchClient(endpoint=endpoint, index_name=semantic_index_name, credential=AzureKeyCredential(key))
    results = client.search(
        search_text=semantic_search_word, 
        include_total_count=True, 
        vectors=[Vector(
            value=response_search_word_vector["data"][0]["embedding"],
            k=3,
            fields="description_vector"
        )], 
        select=["title", "description"],
        query_type="semantic", 
        query_language="ja-JP", 
        semantic_configuration_name="my-semantic-config", 
        query_caption="extractive", 
        query_answer="extractive", 
        top=10
    )
    for result in results:
        print("title:",result["title"])
        print("description:",result["description"])
        captions = result["@search.captions"]
        if captions:
            print("caption:",captions[0])

# ==========================

question = "ミレーについて教えて"

@app.get("/sampleRAG")
async def sampleRAG():
    response = openai.Embedding.create(input=question, engine="sampleEmbedding")
    embeddings = response["data"][0]["embedding"]
    search_client = SearchClient(endpoint=endpoint, index_name=semantic_index_name, credential=AzureKeyCredential(key))
    results = search_client.search(
        search_text=question,
        include_total_count=True,
        vectors=[Vector(
            value=embeddings,
            k=3,
            fields="description_vector"
        )],
        select=["description"],
        top=1
    )
    search_result = []
    for result in results:
        search_result += f"\n・{result['description']}"
    system_prompt = "あなたは優秀なサポートAIです。ユーザーから提供される情報をベースにあなたが学習している情報を付加して回答してください。"
    user_prompt = f"Q: {question}\nA: 参考情報:{search_result}"
    response = openai.ChatCompletion.create(
        engine="sampleChatModel1",
        messages=[
            { "role": "system", "content": system_prompt },
            { "role": "user", "content": user_prompt },
        ],
    )
    text = response["choices"][0]["message"]["content"].replace("\n", "").replace(" .", ".").strip()
    return {"message": text}

# ==========================

# openai.api_baseに相当するもの
AZURE_OPENAI_API_BASE = "https://sample-instance.openai.azure.com/"
AZURE_OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

documents = "Azure OpenAl Serviceは、GPT-4、GPT-35-Turbo、Embeddings モデル シリーズを含むOpenAIの強力な言語モデルへの REST APIアクセスを提供します。さらに、新しい GPT-4 および gpt-35-turboモデル シリーズが一般提供になりました。これらのモデルは、コンテンツ生成、要約、セマンティック検索、自然言語からコードへの変換などを含む特定のタスクに簡単に適合させることができます。ユーザーは、REST API、Python SDK、または Azure OpenAI Studio の Web ベースのインターフェイスを通じてサービスにアクセスできます。"


@app.get("/ragWithLLM")
async def ragWithLLM():
    embeddings = OpenAIEmbeddings(
        openai_api_type="azure",
        model="text-embedding-ada-002",
        openai_api_base=AZURE_OPENAI_API_BASE,
        openai_api_key=AZURE_OPENAI_API_KEY,
        deployment="sampleEmbedding"
    )
    index_name = "langchain-search-demo"
    vector_store = AzureSearch(
        azure_search_endpoint=endpoint,
        azure_search_key=key,
        index_name=index_name,
        embedding_function=embeddings.embed_query
    )
    # なぜかエラーになる。。RuntimeError: Error loading long_text.txt
    # loader = TextLoader("long_text.txt", encoding="utf-8")
    # documents = loader.load()
    text_spliter = CharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0,
    )
    docs = text_spliter.create_documents(documents)
    vector_store.add_documents(docs)
    texts = vector_store.similarity_search(
        query="GPT-4を使ってどんなことができるの？",
        k=3,
        search_type="similarity",
    )
    return {"message": texts[0].page_content}

template = PromptTemplate.from_template("{keyword}を解説する書籍のタイトル案は？")
prompt = template.format(keyword="ChatGPT")

@app.get("/usePromptTemplate")
async def usePromptTemplate():
    chat = AzureChatOpenAI(
        openai_api_base=AZURE_OPENAI_API_BASE,
        openai_api_version="2023-07-01-preview",
        deployment_name="sampleChatModel1",
        openai_api_type="azure",
        )   
    output = chat.predict(prompt)
    return {"message": output}

@app.get("/langChainSample")
async def langChainSample():
    chat = AzureChatOpenAI(
        openai_api_base=AZURE_OPENAI_API_BASE,
        openai_api_version="2023-07-01-preview",
        deployment_name="sampleChatModel1",
        openai_api_key=AZURE_OPENAI_API_KEY,
        openai_api_type="azure",
        )   
    output = chat([
        SystemMessage(content="日本語で回答してください。"),
        HumanMessage(content="ChatGPTについて30文字で教えて。"),
    ])
    return {"message": output}
