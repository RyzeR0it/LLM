import os
import re

def clean_dataset(dataset_path):
  """Cleans a code dataset."""

  with open(dataset_path, 'r', encoding='utf-8') as f:
    data = f.read()

  # Remove any blank lines from the data.
  data = '\n'.join(line for line in data.split('\n') if line.strip())

  # Save the cleaned data to a new file.
  new_dataset_path = dataset_path + '.clean'
  with open(new_dataset_path, 'w', encoding='utf-8') as f:
    f.write(data)


if __name__ == '__main__':
  valid_extensions = ['.txt', '.csv', '.json']  # Add or remove extensions as needed
  for root, dirs, files in os.walk('datasets'):
    for file in files:
      if any(file.endswith(ext) for ext in valid_extensions):
        full_path = os.path.join(root, file)
        clean_dataset(full_path)
