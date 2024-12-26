import pandas as pd
import random

class DNAMutationPreprocessor:
    def __init__(self, input_file, labels_file, output_file):
        """
        Initialize the preprocessor.

        :param input_file: Path to the input DNA mutation file (MAF format).
        :param labels_file: Path to the file containing Hugo Symbols (labels).
        :param output_file: Path to save the processed file (CSV format).
        """
        self.input_file = input_file
        self.labels_file = labels_file
        self.output_file = output_file

    def load_labels(self):
        """
        Load the labels file containing Hugo Symbols.

        :return: List of Hugo Symbols
        """
        labels = pd.read_csv(self.labels_file, sep='\t', header=None)
        labels_list = labels[0].tolist()
        print(f"Loaded {len(labels_list)} labels.")
        return labels_list

    def load_data(self):
        """
        Load the DNA mutation file and create a binary dataframe based on labels.

        :return: Binary DataFrame
        """
        data = pd.read_csv(self.input_file, sep='\t', comment='#', low_memory=False)
        print("Input data shape:", data.shape)
        if 'Hugo_Symbol' not in data.columns or 'IMPACT' not in data.columns:
            raise Exception("Required columns 'Hugo_Symbol' and 'IMPACT' are missing from the input file.")

        # Load labels
        labels = self.load_labels()

        # Create a dictionary for binary values based on 'IMPACT' == 'HIGH'
        binary_dict = {label: 0 for label in labels}
        high_impact_symbols = data.loc[data['IMPACT'] == 'HIGH', 'Hugo_Symbol']

        for symbol in high_impact_symbols:
            if symbol in binary_dict:
                binary_dict[symbol] = 1

        # Convert dictionary to DataFrame
        binary_df = pd.DataFrame(list(binary_dict.items()), columns=['Hugo_Symbol', 'DNA_Mut_values'])
        print("Binary DataFrame shape:", binary_df.shape)
        return binary_df

    def adjust_feature_size(self, data, target_size=17301):
        """
        Adjust the feature size to the target size by randomly removing features with 0 values.

        :param data: DataFrame to process
        :param target_size: Desired number of features
        :return: Adjusted DataFrame
        """
        print("Initial data shape:", data.shape)

        zero_value_indices = data.index[data['DNA_Mut_values'] == 0].tolist()
        excess = data.shape[0] - target_size

        # Randomly drop rows with 0 values to match the target size
        if excess > 0:
            drop_indices = random.sample(zero_value_indices, excess)
            data = data.drop(index=drop_indices)
            print(f"Dropped {len(drop_indices)} rows with 0 values to match target size.")

        print("Final data shape:", data.shape)
        return data

    def save_to_csv(self, data):
        """
        Save the processed data to a CSV file.

        :param data: DataFrame to save
        """
        print("Output data shape:", data.shape)
        data.to_csv(self.output_file, index=False)

    def process(self):
        """
        Perform the entire preprocessing pipeline.
        """
        # Step 1: Load and filter the data
        data = self.load_data()

        # Step 2: Adjust the feature size
        data = self.adjust_feature_size(data)

        # Step 3: Save the final processed data
        self.save_to_csv(data)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess DNA Mutation data for deep learning models.")
    parser.add_argument("input_file", type=str, help="Path to the input DNA Mutation MAF file.")
    parser.add_argument("labels_file", type=str, help="Path to the file containing Hugo Symbols (labels).")
    parser.add_argument("output_file", type=str, help="Path to save the processed CSV file.")

    args = parser.parse_args()

    preprocessor = DNAMutationPreprocessor(args.input_file, args.labels_file, args.output_file)
    preprocessor.process()

# Example usage:
## python DNAMut_preprocess.py wxs.aliquot_ensemble_masked.maf Hugo_symbols.tsv output_file.csv
