from openml.datasets import get_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch

# Load baseline and SwiGLU models
baseline = load_model('baseline_checkpoint.pt')
swiglu = load_model('swiglu_checkpoint.pt')

# Get small OpenML datasets
dataset_ids = [3, 6, 11, 12, 14, 16, 18, 22, 23, 26]  # Small datasets

results = {'baseline': [], 'swiglu': []}

for dataset_id in dataset_ids:
    dataset = get_dataset(dataset_id)
    X, y = dataset.get_data(target=dataset.default_target_attribute)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Evaluate both
    for name, model in [('baseline', baseline), ('swiglu', swiglu)]:
        pred = model.predict(X_train, X_test)
        score = roc_auc_score(y_test, pred)
        results[name].append(score)

# Print results
print(f"Baseline mean: {np.mean(results['baseline']):.4f}")
print(f"SwiGLU mean:   {np.mean(results['swiglu']):.4f}")