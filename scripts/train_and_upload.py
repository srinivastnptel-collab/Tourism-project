import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from huggingface_hub import HfApi, Repository

# 1. Load data
train_path = "mlops/data/processed/train.csv"
valid_path = "mlops/data/processed/valid.csv"

print(f"Loading train data from: {train_path}")
print(f"Loading valid data from: {valid_path}")

train_df = pd.read_csv(train_path)
valid_df = pd.read_csv(valid_path)

# 2. Split features and target

target_col = "ProdTaken"

X_train = train_df.drop(columns=[target_col, "Unnamed: 0"])
y_train = train_df[target_col]

X_valid = valid_df.drop(columns=[target_col, "Unnamed: 0"])
y_valid = valid_df[target_col]

print(f"Train shape: {X_train.shape}, Valid shape: {X_valid.shape}")

# 3. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4. Evaluate
y_pred = model.predict(X_valid)
acc = accuracy_score(y_valid, y_pred)
print(f"Validation Accuracy: {acc:.4f}")

# 5. Save model locally in repo structure
model_dir = "mlops/model"
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, "rf_model.joblib")

joblib.dump(model, model_path)
print(f"Model saved at: {model_path}")

# 6. Upload model to Hugging Face Hub
hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError("HF_TOKEN environment variable is not set. Please configure it in GitHub Secrets.")

repo_id = "Srinivas1969/tourism-rf-model"
api = HfApi()
api.create_repo(repo_id=repo_id, token=hf_token, exist_ok=True)

local_hf_dir = "hf_repo"
repo = Repository(local_dir=local_hf_dir, clone_from=repo_id, use_auth_token=hf_token)

# Save model inside repo
hf_model_path = os.path.join(local_hf_dir, "rf_model.joblib")
joblib.dump(model, hf_model_path)

# Add README
readme_path = os.path.join(local_hf_dir, "README.md")
with open(readme_path, "w") as f:
    f.write(
        "# Tourism Model\n\n"
        "Random Forest classifier trained on the tourism dataset.\n\n"
        f"Validation Accuracy: {acc:.4f}\n"
    )

# Add requirements.txt
with open(os.path.join(local_hf_dir, "requirements.txt"), "w") as f:
    f.write("scikit-learn\njoblib\npandas\n")

# Stage all files and push
repo.git_add()
repo.push_to_hub(commit_message="Upload trained tourism model")
print("✅ Model uploaded to Hugging Face Hub.")
