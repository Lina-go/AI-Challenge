import ollama

res = ollama.chat(
	model="moondream:1.8b",
	messages=[
		{
			'role': 'user',
			'content': 'Describe this image:',
			'images': ['./pdfs/Picture2.png']
		}
	]
)

print(res['message']['content'])