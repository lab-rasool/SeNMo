import pandas as pd

class DNAMethylationPreprocessor:
    def __init__(self, input_file, output_file):
        """
        Initialize the preprocessor.

        :param input_file: Path to the input DNA methylation file (TSV format).
        :param output_file: Path to save the processed file (CSV format).
        """
        self.input_file = input_file
        self.output_file = output_file

    def load_and_transpose(self):
        """
        Load the methylation file and transpose it.

        :return: Transposed DataFrame
        """
        data = pd.read_table(self.input_file)
        print("Input data shape:", data.shape)
        # Transpose data and reset index
        transposed_data = data.transpose()
        transposed_data.reset_index(inplace=True)
        # Ensure column names are correctly set
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
        return data.dropna(axis=1, how='all')  # Drop columns where all values are NaN

    def adjust_feature_size(self, data, target_size=52396):
        """
        Adjust the feature size to the target size by removing features stepwise.

        :param data: DataFrame to process
        :param target_size: Desired number of features
        :return: Adjusted DataFrame
        """
        print("Initial data shape:", data.shape)
        numeric_data = data.iloc[:, 1:].astype(float)  # Assuming first column is sample ID

        # Stepwise removal of features
        intervals = [
            (0, 0),
            (0, 0.1),
            (0.1, 0.2),
            (0.2, 0.3),
            (0.3, 0.4),
            (0.4, 0.5),
            (0.5, 0.6),
            (0.6, 0.7),
            (0.7, 0.8),
            (0.8, 0.9),
            (0.9, 1.0)
        ]

        for lower, upper in intervals:
            if numeric_data.shape[1] > target_size:
                condition = (numeric_data > lower).all() & (numeric_data <= upper).all()
                columns_to_remove = numeric_data.columns[condition]
                to_remove = len(numeric_data.columns) - target_size
                numeric_data = numeric_data.drop(columns=columns_to_remove[:to_remove])
                print(f"After removing features with values ({lower}, {upper}], shape:", numeric_data.shape)

        # Raise exception if target size is not achieved
        if numeric_data.shape[1] > target_size:
            raise Exception(f"Number of features is still greater than {target_size} after all adjustments.")

        # Reconstruct the full DataFrame
        numeric_data.insert(0, 'sample', data.iloc[:, 0])  # Re-add the sample column
        print("Final data shape:", numeric_data.shape)
        return numeric_data

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

    parser = argparse.ArgumentParser(description="Preprocess DNA Methylation data for deep learning models.")
    parser.add_argument("input_file", type=str, help="Path to the input DNA Methylation TSV file.")
    parser.add_argument("output_file", type=str, help="Path to save the processed CSV file.")

    args = parser.parse_args()

    preprocessor = DNAMethylationPreprocessor(args.input_file, args.output_file)
    preprocessor.process()

# Example usage:
## python DNAMethylationPreprocessor.py methylation450.tsv output_file.csv
