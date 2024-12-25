import pandas as pd

class ProteinExpressionPreprocessor:
    def __init__(self, input_file, output_file):
        """
        Initialize the preprocessor.

        :param input_file: Path to the input protein expression file (TSV format).
        :param output_file: Path to save the processed file (CSV format).
        """
        self.input_file = input_file
        self.output_file = output_file

    def load_data(self):
        """
        Load the protein expression file and filter the necessary columns.

        :return: Filtered DataFrame
        """
        data = pd.read_table(self.input_file)
        print("Input data shape:", data.shape)
        if 'AGID' not in data.columns or 'protein_expression' not in data.columns:
            raise Exception("Required columns 'AGID' and 'protein_expression' are missing from the input file.")
        filtered_data = data[['AGID', 'protein_expression']]
        print("Filtered data shape:", filtered_data.shape)
        return filtered_data

    def calculate_mean(self, data):
        """
        Calculate the mean of the 'protein_expression' column.

        :param data: DataFrame to process
        :return: Mean value of 'protein_expression'
        """
        protein_expression_mean = data['protein_expression'].astype(float).mean()
        print(f"Mean value of 'protein_expression': {protein_expression_mean}")
        return protein_expression_mean

    def adjust_feature_size(self, data, target_size=472, mean_value=None):
        """
        Adjust the feature size to the target size by removing and imputing features.

        :param data: DataFrame to process
        :param target_size: Desired number of features
        :param mean_value: Precomputed mean value for 'protein_expression'
        :return: Adjusted DataFrame
        """
        print("Initial data shape:", data.shape)

        # Drop NaN-valued rows randomly until target size is reached
        nan_indices = data.index[data['protein_expression'].isnull()]
        excess = data.shape[0] - target_size
        if excess > 0 and len(nan_indices) > 0:
            drop_indices = nan_indices[:excess]
            data = data.drop(index=drop_indices)
            print(f"Dropped {len(drop_indices)} NaN-valued rows to match target size.")

        # Impute remaining NaNs with the mean of 'protein_expression'
        if 'protein_expression' in data.columns and mean_value is not None:
            data['protein_expression'] = data['protein_expression'].fillna(mean_value)
            print("Imputed missing values in 'protein_expression' with mean value.")

        # Verify no NaN values remain
        if data['protein_expression'].isnull().any():
            raise Exception("There are still missing values after imputation.")

        # Raise exception if target size is still not achieved
        if data.shape[0] > target_size:
            raise Exception(f"Number of features is still greater than {target_size} after all adjustments.")

        print("Final data shape:", data.shape)
        return data

    def transpose_data(self, data):
        """
        Transpose the data for further processing.

        :param data: DataFrame to transpose
        :return: Transposed DataFrame
        """
        transposed_data = data.transpose()
        transposed_data.reset_index(inplace=True)
        transposed_data.columns = transposed_data.iloc[0]
        transposed_data = transposed_data[1:]
        transposed_data.rename(columns={transposed_data.columns[0]: 'sample'}, inplace=True)
        return transposed_data

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

        # Step 2: Calculate the mean of 'protein_expression'
        mean_value = self.calculate_mean(data)

        # Step 3: Adjust the feature size and impute missing values
        data = self.adjust_feature_size(data, mean_value=mean_value)

        # Step 4: Transpose the data
        data = self.transpose_data(data)

        # Step 5: Save the final processed data
        self.save_to_csv(data)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Protein Expression data for deep learning models.")
    parser.add_argument("input_file", type=str, help="Path to the input Protein Expression TSV file.")
    parser.add_argument("output_file", type=str, help="Path to save the processed CSV file.")

    args = parser.parse_args()s

    preprocessor = ProteinExpressionPreprocessor(args.input_file, args.output_file)
    preprocessor.process()

# Example usage:
## python ProteinExpresn_preprocess.py sample_RPPA_data.tsv output_file.csv
