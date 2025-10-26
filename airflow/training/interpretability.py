import shap
import matplotlib.pyplot as plt

def generate_shap_plots(model, X_sample, output_dir, max_samples=500):
    """
    Genera gráficos SHAP y los guarda en el directorio especificado.
    
    Args:
        model: Pipeline con preprocesador y clasificador
        X_sample: Muestra de datos para explicar
        output_dir: Directorio donde guardar los gráficos
        max_samples: Número máximo de muestras a usar
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    clf = model.named_steps['clf']
    X_transformed = model.named_steps['prep'].transform(X_sample[:max_samples])
    
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_transformed)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    # Gráfico de resumen
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_transformed, show=False)
    summary_path = os.path.join(output_dir, "shap_summary.png")
    plt.savefig(summary_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    # Gráfico de importancia
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_transformed, plot_type="bar", show=False)
    importance_path = os.path.join(output_dir, "shap_importance.png")
    plt.savefig(importance_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    return summary_path, importance_path
