import torch
from load_datasets import load_and_preprocess_data

def evaluate_model(model, dataset):
  """Evaluates the Transformer model."""

  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters())

  model.eval()
  with torch.no_grad():
    data, targets = dataset
    output = model(data)
    loss = loss_fn(output, targets)
    _, predicted = torch.max(output, 1)
    correct = (predicted == targets).sum().item()
    total = targets.size(0)
    accuracy = correct / total

  print('Loss:', loss.item())
  print('Accuracy:', accuracy)

def main():
  # Load the data and vocab
  data, vocab, text_pipeline, label_pipeline = load_and_preprocess_data()

  # Load the model
  model = torch.load('models/Transformer.pt')

  # Set the model to evaluation mode
  model.eval()

  evaluate_model(model, data)

if __name__ == '__main__':
  main()
