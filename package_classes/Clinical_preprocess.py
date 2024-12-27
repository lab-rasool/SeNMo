import pandas as pd

class ClinicalDataPreprocessor:
    def __init__(self, input_file, output_file):
        """
        Initialize the preprocessor.

        :param input_file: Path to the input clinical data file (TSV format).
        :param output_file: Path to save the processed file (CSV format).
        """
        self.input_file = input_file
        self.output_file = output_file

    def load_data(self):
        """
        Load the clinical data file and filter the required columns.

        :return: Filtered DataFrame
        """
        data = pd.read_csv(self.input_file, sep='\t', low_memory=False)
        print("Input data shape:", data.shape)

        required_columns = {
            'age_at_index.demographic': 'age',
            'gender.demographic': 'gender',
            'race.demographic': 'race',
            'tumor_stage.diagnoses': 'stage'
        }

        for col in required_columns:
            if col not in data.columns:
                raise Exception(f"Required column '{col}' is missing from the input file.")

        # Rename columns
        data = data[list(required_columns.keys())]
        data.rename(columns=required_columns, inplace=True)

        print("Filtered and renamed data shape:", data.shape)
        return data

    def convert_categorical_to_numerical(self, data):
        """
        Convert categorical columns to numerical values.

        :param data: DataFrame to process
        :return: DataFrame with converted columns
        """
        # Convert 'age' to float
        data['age'] = data['age'].astype(float)

        # Map values for 'gender'
        gender_map = {'male': 1, 'female': 2}
        data['gender'] = data['gender'].map(gender_map)

        # Map values for 'race'
        race_map = {
            'white': 1,
            'asian': 2,
            'black or african american': 3,
            'not reported': 4,
            'american indian or alaska native': 5
        }
        data['race'] = data['race'].map(race_map)

        # Map values for 'stage'
        stage_map = {
            'stage 0': 1,
            'is': 10, 'stage i': 10,
            'stage ia': 11, 'stage ib': 12, 'stage ic': 13,
            'stage ii': 20, 'i/ii nos': 20,
            'stage iia': 21, 'stage iib': 22, 'stage iic': 23,
            'stage iii': 30, 'stage iiia': 31, 'stage iiib': 32, 'stage iiic': 33,
            'stage iv': 40, 'stage iva': 41, 'stage ivb': 42, 'stage ivc': 43,
            'not reported': 50, 'stage x': 50
        }
        data['stage'] = data['stage'].map(stage_map)

        print("Converted categorical data to numerical values.")
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

        # Step 2: Convert categorical columns to numerical values
        data = self.convert_categorical_to_numerical(data)

        # Step 3: Save the final processed data
        self.save_to_csv(data)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Clinical data for deep learning models.")
    parser.add_argument("input_file", type=str, help="Path to the input clinical data TSV file.")
    parser.add_argument("output_file", type=str, help="Path to save the processed CSV file.")

    args = parser.parse_args()

    preprocessor = ClinicalDataPreprocessor(args.input_file, args.output_file)
    preprocessor.process()

# Example usage:
## python Clinical_preprocess.py GDC_phenotype.tsv output_file.csv
