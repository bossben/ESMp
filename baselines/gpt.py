import openai
import time
import anthropic

openai.api_key = 'KEY'
claude_key = 'KEY'




gpt_model = "gpt-4-0125-preview" # gpt 4 turbo
claude_model = "claude-3-opus-20240229" # claude 3 opus
client = anthropic.Anthropic(
    api_key=claude_key,
)


def claudecall(system,user,history=None):
	return "SELECT "
	while(True):
		try:
			messages = []
			if(history):
				for h in history:
					messages.append(h)
			messages.append({"role": "user", "content": 
						user})
			response = client.messages.create(
				model=claude_model,
				max_tokens = 1000,
				temperature=0,
				system=system,
				messages=messages
			)
			break
		except Exception as e:
			print(e)
			time.sleep(1)
	return response.content[0].text

def gptcall(system,user,history=None):
	return "SELECT "
	while(True):
		try:
			messages = [
				{"role": "system", "content": system}
			]
			if(history):
				for h in history:
					messages.append(h)
			messages.append({"role": "user", "content": user})
			response = openai.ChatCompletion.create(
				model=gpt_model,
				temperature=0,
				messages=messages
			)
			break
		except Exception as e:
			print(e)
			time.sleep(1)

	r = response['choices'][0]['message']['content']
	return str(r)