import pickle
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

class SeNMoInference:
    def __init__(self, model_class, checkpoint_dir, input_file, output_dir):
        """
        Initialize the SeNMo Inference class.

        :param model_class: Class of the trained model.
        :param checkpoint_dir: Directory containing the 10 checkpoints.
        :param input_file: Path to the .pkl file containing the sample data.
        :param output_dir: Directory to save the output embeddings and hazard scores.
        """
        self.model_class = model_class
        self.checkpoint_dir = Path(checkpoint_dir)
        self.input_file = input_file
        self.output_dir = Path(output_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_data(self):
        """
        Load the sample data from the input .pkl file.

        :return: x_omic feature vector from the input data.
        """
        with open(self.input_file, 'rb') as f:
            data = pickle.load(f)
        x_omic = np.array(data['cv_splits'][1]['test']['x_omic'][0], dtype=np.float32)
        return torch.tensor(x_omic, dtype=torch.float32).unsqueeze(0).to(self.device)

    def load_checkpoints(self):
        """
        Load all the model checkpoints.

        :return: List of loaded model instances.
        """
        checkpoints = list(self.checkpoint_dir.glob("*.pt"))
        models = []
        for checkpoint in checkpoints:
            model = self.model_class()
            checkpoint_data = torch.load(checkpoint, map_location=self.device)
            state_dict = checkpoint_data['model_state_dict']

            # Remove 'module.' prefix if present (from DataParallel)
            if any(key.startswith("module.") for key in state_dict.keys()):
                state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}

            # Remove unexpected keys
            state_dict = {k: v for k, v in state_dict.items() if k not in ["output_range", "output_shift"]}

            # Adjust state_dict keys if necessary
            if "classifier.0.weight" in state_dict:
                state_dict["classifier.weight"] = state_dict.pop("classifier.0.weight")
                state_dict["classifier.bias"] = state_dict.pop("classifier.0.bias")

            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            models.append(model)
        return models

    def generate_embeddings(self):
        """
        Generate embeddings and hazard scores using the model checkpoints.

        :return: Aggregated embeddings and hazard scores.
        """
        x_omic = self.load_data()
        models = self.load_checkpoints()

        all_embeddings = []
        all_hazard_scores = []

        for model in models:
            with torch.no_grad():
                features = model.encoder(x_omic)  # Extract features from encoder
                hazard_score = model.classifier(features)  # Perform regression
                all_embeddings.append(features.cpu().numpy())
                all_hazard_scores.append(hazard_score.cpu().item())

        aggregated_embedding = np.mean(np.vstack(all_embeddings), axis=0).astype(np.float32)
        aggregated_hazard_score = float(np.mean(all_hazard_scores))

        return aggregated_embedding, aggregated_hazard_score

    def save_outputs(self, embedding, hazard_score):
        """
        Save the embeddings and hazard scores to a .pkl file.

        :param embedding: Aggregated 48-dimensional embedding.
        :param hazard_score: Aggregated hazard score.
        """
        output_file = self.output_dir / f"{Path(self.input_file).stem}_outputs.pkl"
        output_data = {
            "embedding": embedding,
            "hazard_score": hazard_score
        }
        with open(output_file, 'wb') as f:
            pickle.dump(output_data, f)
        print(f"Saved outputs to {output_file}")

    def process(self):
        """
        Run the entire inference pipeline.
        """
        embedding, hazard_score = self.generate_embeddings()
        self.save_outputs(embedding, hazard_score)

if __name__ == "__main__":
    import argparse

    class SeNMo(nn.Module):
        def __init__(self, input_dim=80697, hidden_layers=[1024, 512, 256, 128, 48, 48, 48], dropout_rate=0.25, label_dim=1):
            super(SeNMo, self).__init__()
            layers = []

            prev_dim = input_dim
            for hidden_dim in hidden_layers:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ELU())
                layers.append(nn.AlphaDropout(p=dropout_rate))
                prev_dim = hidden_dim

            self.encoder = nn.Sequential(*layers)
            self.classifier = nn.Linear(prev_dim, label_dim)  # Regression output

        def forward(self, x):
            features = self.encoder(x)
            out = self.classifier(features)
            return features, out

    parser = argparse.ArgumentParser(description="Generate embeddings and hazard scores using trained SeNMo models.")
    parser.add_argument("checkpoint_dir", type=str, help="Path to the directory containing model checkpoints.")
    parser.add_argument("input_file", type=str, help="Path to the .pkl file containing the sample data.")
    parser.add_argument("output_dir", type=str, help="Directory to save the output embeddings and hazard scores.")

    args = parser.parse_args()

    inference = SeNMoInference(
        model_class=SeNMo,
        checkpoint_dir=args.checkpoint_dir,
        input_file=args.input_file,
        output_dir=args.output_dir
    )
    inference.process()

# Example usage:
# python generate_embeddings.py ./checkpoints ./multiomic_features.pkl ./outputs
