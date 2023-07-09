import os
import shutil
import time
from tqdm import tqdm
import torch

import clean_datasets
import filter_datasets
import load_datasets
import train_model
import evaluate_model
import generate_code
from generate_code import main


def main():
    # Get a list of all dataset files
    dataset_files = [f for f in os.listdir('datasets') if os.path.isfile(os.path.join('datasets', f))]

    # Create a progress bar
    progress_bar = tqdm(total=len(dataset_files), desc='Processing datasets', dynamic_ncols=True)

    for file in dataset_files:
        # Clean the dataset
        clean_datasets.clean_dataset(os.path.join('datasets', file))
        progress_bar.update()

        # Filter the dataset
        filter_datasets.filter_dataset(os.path.join('datasets', file + '.clean'))
        progress_bar.update()

        # Move the processed file to a different directory
        shutil.move(os.path.join('datasets', file), os.path.join('datasets_done', file))

    # Load and preprocess the data
    data, vocab, text_pipeline, label_pipeline = load_datasets.load_and_preprocess_data()

    # Train the model
    model = train_model.train_model(data, vocab, text_pipeline, label_pipeline)

    # Evaluate the model
    evaluate_model.evaluate_model(model, data)

    # Generate some code
    generated_code = generate_code.generate_code(model, text_pipeline, vocab)
    print('Generated code:\n', generated_code)


if __name__ == '__main__':
    main()