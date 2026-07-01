from openml.datasets import get_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch
import json
import numpy as np

from model import NanoTabPFNModel

def load_model(checkpoint_path, ffn_type='gelu'):
    model = NanoTabPFNModel(
        embedding_size=96,
        num_attention_heads=4,
        mlp_hidden_size=192,
        num_layers=3,
        num_outputs=2,
        ffn_type=ffn_type
    )
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()
    classifier = NanoTabPFNClassifier(model, torch.device(device))
    return classifier

# Load baseline and SwiGLU models
baseline = load_model('model_checkpoints/gelu_checkpoint.pt', ffn_type='gelu')
swiglu = load_model('model_checkpoints/swiglu_checkpoint.pt', ffn_type='swiglu')

# Get small OpenML datasets
dataset_ids = [3, 6, 11, 12, 14, 16, 18, 22, 23, 26]  # Small datasets

results = {'baseline': [], 'swiglu': []}

for dataset_id in dataset_ids:
    dataset = get_dataset(dataset_id)
    X, y, _, _  = dataset.get_data(target=dataset.default_target_attribute)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Evaluate both
    for name, classifier in [('baseline', baseline), ('swiglu', swiglu)]:
        classifier.fit(X_train, y_train)
        pred = classifier.predict(X_test)
        score = roc_auc_score(y_test, pred)
        results[name].append(score)

# Print results
print(f"Baseline mean: {np.mean(results['baseline']):.4f}")
print(f"SwiGLU mean:   {np.mean(results['swiglu']):.4f}")

results_summary = {
    'baseline_mean': float(np.mean(results['baseline'])),
    'baseline_std': float(np.std(results['baseline'])),
    'swiglu_mean': float(np.mean(results['swiglu'])),
    'swiglu_std': float(np.std(results['swiglu'])),
    'improvement': float(np.mean(results['swiglu']) - np.mean(results['baseline'])),
    'dataset_ids': dataset_ids,
    'baseline_scores': results['baseline'],
    'swiglu_scores': results['swiglu']
}

# Save to JSON
with open('openml_evaluation_results.json', 'w') as f:
    json.dump(results_summary, f, indent=2)

print("\nResults saved to openml_evaluation_results.json")
