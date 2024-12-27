import pandas as pd

class MiRNAPreprocessor:
    def __init__(self, input_file, output_file):
        """
        Initialize the preprocessor.

        :param input_file: Path to the input miRNA file (TSV format).
        :param output_file: Path to save the processed file (CSV format).
        """
        self.input_file = input_file
        self.output_file = output_file

    def load_and_transpose(self):
        """
        Load the miRNA file and transpose it.

        :return: Transposed DataFrame
        """
        data = pd.read_table(self.input_file)
        transposed_data = data.transpose()
        transposed_data.reset_index(inplace=True)
        transposed_data.columns = transposed_data.iloc[0]
        transposed_data = transposed_data[1:]
        transposed_data.rename(columns={transposed_data.columns[0]: 'sample'}, inplace=True)
        return transposed_data

    def drop_nans(self, data):
        """
        Drop rows with NaN values.

        :param data: DataFrame to process
        :return: DataFrame with NaN values dropped
        """
        return data.dropna()

    def adjust_feature_size(self, data, target_size=1731):
        """
        Ensure the number of features is equal to the target size by dropping zero-value columns if needed.

        :param data: DataFrame to process
        :param target_size: Desired number of features
        :return: Adjusted DataFrame
        """
        if data.shape[1] > target_size:
            # Identify columns with all zero values
            zero_columns = data.columns[(data == 0).all()]
            data = data.drop(columns=zero_columns[:data.shape[1] - target_size])
        return data

    def save_to_csv(self, data):
        """
        Save the processed data to a CSV file.

        :param data: DataFrame to save
        """
        data.to_csv(self.output_file, index=False)

    def process(self):
        """
        Perform the entire preprocessing pipeline.
        """
        # Step 1: Load and transpose the data
        data = self.load_and_transpose()

        # Step 2: Drop NaN values
        data = self.drop_nans(data)

        # Step 3: Adjust the feature size if necessary
        data = self.adjust_feature_size(data)

        # Step 4: Save the final processed data
        self.save_to_csv(data)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess miRNA data for deep learning models.")
    parser.add_argument("input_file", type=str, help="Path to the input miRNA TSV file.")
    parser.add_argument("output_file", type=str, help="Path to save the processed CSV file.")

    args = parser.parse_args()

    preprocessor = MiRNAPreprocessor(args.input_file, args.output_file)
    preprocessor.process()

# Example usage:
## python miRNA_preprocess.py input_mirna.tsv output_mirna.csv
