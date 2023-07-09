import os
import re

def filter_dataset(dataset_path):
  """Filters a code dataset."""

  with open(dataset_path, 'r') as f:
    data = f.read()

  # Modify these as per your requirement
  data = re.sub(r'def function_name\(.*?\):', '', data)
  data = re.sub(r'import.*?\n', '', data)

  # Save the filtered data to a new file.
  new_dataset_path = os.path.splitext(dataset_path)[0] + '.filtered'
  with open(new_dataset_path, 'w') as f:
    f.write(data)

if __name__ == '__main__':
  directory = 'datasets'
  for filename in os.listdir(directory):
    if filename.endswith('.clean'):
      filter_dataset(os.path.join(directory, filename))
