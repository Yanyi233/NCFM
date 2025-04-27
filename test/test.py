# import requests

# url = "https://api.x.ai/v1/chat/completions"
# api_key = "xai-6mm4IQI5MOiJhwo6JCJ6mFrQuJSRqpjM6LWn22pMZJGbhMgf9V4LkOL09k9F97vu3wH59qINrnhdG6T3"  # 替换为你的实际API密钥

# headers = {
#     "Content-Type": "application/json",
#     "Authorization": f"Bearer {api_key}"
# }

# data = {
#     "messages": [
#         {
#             "role": "system",
#             "content": "You are a test assistant."
#         },
#         {
#             "role": "user",
#             "content": "Testing. Just say hi and hello world and nothing else."
#         }
#     ],
#     "model": "grok-3-latest",
#     "stream": False,
#     "temperature": 0
# }

# response = requests.post(url, headers=headers, json=data)

# # 打印响应
# print(response.status_code)
# print(response.json())

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version built with PyTorch: {torch.version.cuda}")
    print(f"Number of GPUs detected by PyTorch: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("PyTorch cannot find or use CUDA GPUs.")