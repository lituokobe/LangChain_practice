from langserve import RemoteRunnable

if __name__ == '__main__':
    client = RemoteRunnable('http://localhost:8000/chainDemo/')
    print(client.invoke({'language': 'Chinese', 'text': 'Isn\'t this a beautiful day?'}))