import shap
import matplotlib.pyplot as plt
import mlflow

def log_shap_to_mlflow(model, X_sample, max_samples=500):
    
    clf = model.named_steps['clf']
    X_transformed = model.named_steps['prep'].transform(X_sample[:max_samples])
    
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_transformed)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_transformed, show=False)
    summary_path = "/tmp/shap_summary.png"
    plt.savefig(summary_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_transformed, plot_type="bar", show=False)
    importance_path = "/tmp/shap_importance.png"
    plt.savefig(importance_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    mlflow.log_artifact(summary_path, "interpretability")
    mlflow.log_artifact(importance_path, "interpretability")
