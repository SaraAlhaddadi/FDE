import cohere
import os

co = cohere.Client(os.getenv("COHERE_API_KEY"))
message = "What is Machine Learning?"
response = co.chat(
  message=message,
  model="command-r-plus-08-2024",
  temperature=0.3
)


answer = response.text
print(answer)