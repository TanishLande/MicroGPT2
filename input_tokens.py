import tiktoken

with open('input.txt', 'r') as f:
            text = f.read()
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode(text)

num_lens = len(tokens)
print(f"Number of tokens: {num_lens}")  