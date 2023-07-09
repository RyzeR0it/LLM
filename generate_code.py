import torch
from load_datasets import load_and_preprocess_data

def generate_code(prompt, model, tokenizer, vocab, device, max_length=100):
  """Generates code based on a prompt."""

  # Tokenize the prompt
  tokenized_prompt = [vocab.stoi[word] for word in tokenizer(prompt)]
  tokenized_prompt = torch.tensor(tokenized_prompt, dtype=torch.long, device=device).unsqueeze(0)

  generated_code = []

  # Generate code
  for _ in range(max_length):
    # Predict the next token
    with torch.no_grad():
      output = model(tokenized_prompt)
      predicted_token = output.argmax(dim=-1)[:, -1]

    # Append the predicted token to the prompt
    tokenized_prompt = torch.cat([tokenized_prompt, predicted_token.unsqueeze(0)], dim=-1)

    # Append the predicted token to the generated code
    generated_code.append(predicted_token.item())

    # If the predicted token is the end-of-sequence token, stop
    if predicted_token.item() == vocab.stoi['<eos>']:
      break

  # Detokenize the generated code
  generated_code = ' '.join(vocab.itos[token] for token in generated_code)

  return generated_code

def main():
    """Prompts the user to enter a prompt and then generates code based on the prompt."""

    model = torch.load('models/Transformer.pt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, vocab, text_pipeline, label_pipeline = load_and_preprocess_data()
    prompt = input("Enter a prompt: ")
    generated_code = generate_code(prompt, model, text_pipeline, vocab, device)
    print(generated_code)

if __name__ == '__main__':
    main()