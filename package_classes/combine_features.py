# import pandas as pd
# import numpy as np
# import pickle

# class DataCombiner:
#     def __init__(self, clinical_file, dnamut_file, protein_file, gene_file, methylation_file, mirna_file, output_file):
#         """
#         Initialize the combiner with file paths.

#         :param clinical_file: Path to the pre-processed clinical data file.
#         :param dnamut_file: Path to the pre-processed DNA mutation data file.
#         :param protein_file: Path to the pre-processed protein expression data file.
#         :param gene_file: Path to the pre-processed gene expression data file.
#         :param methylation_file: Path to the pre-processed DNA methylation data file.
#         :param mirna_file: Path to the pre-processed miRNA expression data file.
#         :param output_file: Path to save the combined data object as a .pkl file.
#         """
#         self.files = {
#             'clinical': (clinical_file, 4),
#             'dnamut': (dnamut_file, 17301),
#             'protein': (protein_file, 472),
#             'gene': (gene_file, 8794),
#             'methylation': (methylation_file, 52396),
#             'mirna': (mirna_file, 1730)
#         }
#         self.output_file = output_file

#     def load_data(self, file_path, expected_dim, key):
#         """
#         Load a data file and handle specific cases like clinical and DNA mutation data.

#         :param file_path: Path to the data file.
#         :param expected_dim: Expected dimension of the data.
#         :param key: Key to identify the type of data being loaded.
#         :return: Loaded data as a numpy array.
#         """
#         try:
#             data = pd.read_csv(file_path, header=None)

#             # Handle clinical data with two rows
#             if key == 'clinical' and data.shape[0] == 2:
#                 data = data.iloc[1, :]
#                 print(f"Using the second row for clinical data from {file_path}.")

#             # Handle DNA mutation data with extra dimensions
#             elif key == 'dnamut' and data.shape[0] == 2:
#                 data = data.iloc[1, :]

#             # Handle protein data with two rows
#             elif key == 'protein' and data.shape[0] == 2:
#                 if 'sample' in data.iloc[:, 0].values or 'protein_expression' in data.iloc[:, 0].values:
#                     data = data.iloc[1, 1:]
#                 else:
#                     data = data.iloc[1, :]

#             # Handle gene expression data with two rows
#             elif key == 'gene' and data.shape[0] == 2:
#                 if 'sample' in data.iloc[:, 0].values or 'gene_expression' in data.iloc[:, 0].values:
#                     data = data.iloc[1, 1:]
#                 else:
#                     data = data.iloc[1, :]

#             # Handle DNA methylation data with two rows
#             elif key == 'methylation' and data.shape[0] == 2:
#                 data = data.iloc[1, :]
#                 print(f"Loaded data from {file_path} with shape {data.shape}.")

#             # Handle miRNA data with two rows
#             elif key == 'mirna' and data.shape[0] == 2:
#                 data = data.iloc[1, :]
#                 print(f"Loaded data from {file_path} with shape {data.shape}.")

#             # Exclude potential row headers or names
#             if len(data.shape) > 1 and data.shape[1] > expected_dim:
#                 data = data.iloc[:, 1:]
#             elif len(data.shape) > 1 and data.shape[0] > expected_dim:
#                 data = data.iloc[:expected_dim, :]
#             elif len(data.shape) == 1 and data.shape[0] > expected_dim:
#                 data = data.iloc[:expected_dim]

#             print(f"Loaded data from {file_path} with shape {data.shape}.")
#             return data.values.flatten()
#         except FileNotFoundError:
#             print(f"File {file_path} not found. Using zero vector.")
#             return np.zeros(expected_dim)
#         except IndexError:
#             print(f"Data from {file_path} does not have the expected dimensions. Using zero vector.")
#             return np.zeros(expected_dim)

#     def combine_features(self):
#         """
#         Combine all data modalities into a single feature vector.

#         :return: Combined feature vector.
#         """
#         combined_vector = []
#         for key, (file_path, dim) in self.files.items():
#             feature_vector = self.load_data(file_path, dim, key)
#             if len(feature_vector) != dim:
#                 raise ValueError(f"Unexpected dimension for {key}. Expected {dim}, got {len(feature_vector)}.")
#             combined_vector.append(feature_vector)
#         combined_vector = np.concatenate(combined_vector)
#         if len(combined_vector) != 80697:
#             raise ValueError(f"Final concatenated vector has incorrect dimension: {len(combined_vector)}")
#         print("Combined feature vector dimension:", combined_vector.shape)
#         return combined_vector

#     def save_as_pkl(self, combined_vector):
#         """
#         Save the combined feature vector in the specified format.

#         :param combined_vector: Combined feature vector.
#         """
#         data = {
#             'cv_splits': {
#                 1: {
#                     'test': {
#                         'x_patname': ['test_sample'],
#                         'x_omic': [combined_vector]
#                     }
#                 }
#             }
#         }
#         with open(self.output_file, 'wb') as f:
#             pickle.dump(data, f)
#         print(f"Saved combined data to {self.output_file}.")

#     def process(self):
#         """
#         Perform the entire process of combining features and saving them.
#         """
#         combined_vector = self.combine_features()
#         self.save_as_pkl(combined_vector)

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Combine preprocessed data modalities into a single feature vector.")
#     parser.add_argument("clinical_file", type=str, help="Path to the clinical data file.")
#     parser.add_argument("dnamut_file", type=str, help="Path to the DNA mutation data file.")
#     parser.add_argument("protein_file", type=str, help="Path to the protein expression data file.")
#     parser.add_argument("gene_file", type=str, help="Path to the gene expression data file.")
#     parser.add_argument("methylation_file", type=str, help="Path to the DNA methylation data file.")
#     parser.add_argument("mirna_file", type=str, help="Path to the miRNA expression data file.")
#     parser.add_argument("output_file", type=str, help="Path to save the combined data object as a .pkl file.")

#     args = parser.parse_args()

#     combiner = DataCombiner(
#         args.clinical_file,
#         args.dnamut_file,
#         args.protein_file,
#         args.gene_file,
#         args.methylation_file,
#         args.mirna_file,
#         args.output_file
#     )
#     combiner.process()

# # Example usage:
# # python combine_data.py clinical.csv dnamut.csv protein.csv gene.csv methylation.csv mirna.csv combined_data.pkl

import pandas as pd
import numpy as np
import pickle

class DataCombiner:
    def __init__(self, clinical_file, dnamut_file, protein_file, gene_file, methylation_file, mirna_file, output_file):
        """
        Initialize the combiner with file paths.

        :param clinical_file: Path to the pre-processed clinical data file.
        :param dnamut_file: Path to the pre-processed DNA mutation data file.
        :param protein_file: Path to the pre-processed protein expression data file.
        :param gene_file: Path to the pre-processed gene expression data file.
        :param methylation_file: Path to the pre-processed DNA methylation data file.
        :param mirna_file: Path to the pre-processed miRNA expression data file.
        :param output_file: Path to save the combined data object as a .pkl file.
        """
        self.files = {
            'clinical': (clinical_file, 4),
            'dnamut': (dnamut_file, 17301),
            'protein': (protein_file, 472),
            'gene': (gene_file, 8794),
            'methylation': (methylation_file, 52396),
            'mirna': (mirna_file, 1730)
        }
        self.output_file = output_file

    def load_data(self, file_path, expected_dim, key):
        """
        Load a data file and handle specific cases like clinical and DNA mutation data.

        :param file_path: Path to the data file.
        :param expected_dim: Expected dimension of the data.
        :param key: Key to identify the type of data being loaded.
        :return: Loaded data as a numpy array.
        """
        try:
            data = pd.read_csv(file_path, header=None)

            # Handle clinical data with two rows
            if key == 'clinical' and data.shape[0] == 2:
                data = data.iloc[1, :]
                # print(f"Using the second row for clinical data from {file_path}.")

            # Handle DNA mutation data with extra dimensions
            elif key == 'dnamut' and data.shape[0] == 2:
                data = data.iloc[1, :]

            # Handle protein data with two rows
            elif key == 'protein' and data.shape[0] == 2:
                if 'sample' in data.iloc[:, 0].values or 'protein_expression' in data.iloc[:, 0].values:
                    data = data.iloc[1, 1:]
                else:
                    data = data.iloc[1, :]

            # Handle gene expression data with two rows
            elif key == 'gene' and data.shape[0] == 2:
                if 'sample' in data.iloc[:, 0].values or 'gene_expression' in data.iloc[:, 0].values:
                    data = data.iloc[1, 1:]
                else:
                    data = data.iloc[1, :]

            # Handle DNA methylation data with two rows
            elif key == 'methylation' and data.shape[0] == 2:
                data = data.iloc[1, :]
                # print(f"Loaded data from {file_path} with shape {data.shape}.")

            # Handle miRNA data with two rows
            elif key == 'mirna' and data.shape[0] == 2:
                data = data.iloc[1, :]
                # print(f"Loaded data from {file_path} with shape {data.shape}.")

            # Exclude potential row headers or names
            if len(data.shape) > 1 and data.shape[1] > expected_dim:
                data = data.iloc[:, 1:]
            elif len(data.shape) > 1 and data.shape[0] > expected_dim:
                data = data.iloc[:expected_dim, :]
            elif len(data.shape) == 1 and data.shape[0] > expected_dim:
                data = data.iloc[:expected_dim]

            print(f"Loaded data from {file_path} with shape {data.shape}.")
            return data.values.flatten()
        except FileNotFoundError:
            print(f"File {file_path} not found. Using zero vector.")
            return np.zeros(expected_dim)
        except IndexError:
            print(f"Data from {file_path} does not have the expected dimensions. Using zero vector.")
            return np.zeros(expected_dim)

    def combine_features(self):
        """
        Combine all data modalities into a single feature vector.

        :return: Combined feature vector.
        """
        combined_vector = []
        for key, (file_path, dim) in self.files.items():
            feature_vector = self.load_data(file_path, dim, key)
            if len(feature_vector) != dim:
                raise ValueError(f"Unexpected dimension for {key}. Expected {dim}, got {len(feature_vector)}.")
            combined_vector.append(feature_vector)
        combined_vector = np.concatenate(combined_vector)
        if len(combined_vector) != 80697:
            raise ValueError(f"Final concatenated vector has incorrect dimension: {len(combined_vector)}")
        print("Combined feature vector dimension:", combined_vector.shape)
        return combined_vector

    def save_as_pkl(self, combined_vector):
        """
        Save the combined feature vector in the specified format.

        :param combined_vector: Combined feature vector.
        """
        # Check for numpy.object_ type elements in the combined vector
        if np.any([isinstance(el, np.object_) for el in combined_vector]):
            raise TypeError("The combined vector contains elements of type numpy.object_ which are not allowed.")

        data = {
            'cv_splits': {
                1: {
                    'test': {
                        'x_patname': ['test_sample'],
                        'x_omic': [combined_vector]
                    }
                }
            }
        }
        with open(self.output_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved combined data to {self.output_file}.")

    def process(self):
        """
        Perform the entire process of combining features and saving them.
        """
        combined_vector = self.combine_features()
        # Check for numpy.object_ type elements before saving
        if np.any([isinstance(el, np.object_) for el in combined_vector]):
            raise TypeError("The combined vector contains elements of type numpy.object_ which are not allowed.")
        self.save_as_pkl(combined_vector)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Combine preprocessed data modalities into a single feature vector.")
    parser.add_argument("clinical_file", type=str, help="Path to the clinical data file.")
    parser.add_argument("dnamut_file", type=str, help="Path to the DNA mutation data file.")
    parser.add_argument("protein_file", type=str, help="Path to the protein expression data file.")
    parser.add_argument("gene_file", type=str, help="Path to the gene expression data file.")
    parser.add_argument("methylation_file", type=str, help="Path to the DNA methylation data file.")
    parser.add_argument("mirna_file", type=str, help="Path to the miRNA expression data file.")
    parser.add_argument("output_file", type=str, help="Path to save the combined data object as a .pkl file.")

    args = parser.parse_args()

    combiner = DataCombiner(
        args.clinical_file,
        args.dnamut_file,
        args.protein_file,
        args.gene_file,
        args.methylation_file,
        args.mirna_file,
        args.output_file
    )
    combiner.process()

# Example usage:
# python combine_features.py clinical.csv dnamut.csv protein.csv gene.csv methylation.csv mirna.csv multiomic_features.pkl

