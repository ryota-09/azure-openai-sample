from fastapi import FastAPI
import os
import openai
from azure.core.credentials import AzureKeyCredential
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

