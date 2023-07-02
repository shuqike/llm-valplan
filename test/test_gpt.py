import openai


def get_api(file='keys/openai_1'):
    with open(file, 'r') as f:
        openai.api_key = f.read()


if __name__ == '__main__':
    get_api()
    chat_completion = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{"role": "user", "content": "Hello world"}]
    )
