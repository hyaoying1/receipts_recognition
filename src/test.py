from openai import OpenAI

client = OpenAI(
    api_key="sk-88551cce573d49fe81aa466d78c21741",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

models = client.models.list()

print("Available models:")
for m in models.data:
    print("-", m.id)