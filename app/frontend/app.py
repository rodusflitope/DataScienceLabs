import gradio as gr
import requests
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

def predict_individual(customer_id, product_id):
    try:
        response = requests.post(
            f"{BACKEND_URL}/predict",
            json={
                "customer_id": int(customer_id),
                "product_id": int(product_id)
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            prediction = "SÍ comprará" if result["prediction"] == 1 else "NO comprará"
            probability = result["prediction_proba"] * 100
            
            return f"""
## Predicción Individual

**{prediction}**

**Probabilidad:** {probability:.1f}%

---
Cliente ID: {result['customer_id']} | Producto ID: {result['product_id']} | Semana predicha: {result['week']}
            """
        else:
            error = response.json().get('detail', 'Error desconocido')
            return f"Error: {error}"
    
    except Exception as e:
        return f"Error de conexión: {str(e)}"

def predict_aggregate(product_id):
    try:
        response = requests.post(
            f"{BACKEND_URL}/predict/aggregate",
            json={
                "product_id": int(product_id)
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            percentage = result["expected_percentage"]
            
            return f"""
## Predicción Agregada

**Producto:** {result['product_name']}

**Total de clientes:** {result['total_customers']}

**Clientes que comprarán:** {result['expected_buyers']} ({percentage:.1f}%)

---
Producto ID: {result['product_id']} | Semana predicha: {result['week']}
            """
        else:
            error = response.json().get('detail', 'Error desconocido')
            return f"Error: {error}"
    
    except Exception as e:
        return f"Error de conexión: {str(e)}"

with gr.Blocks(title="SodAI Drinks") as app:
    gr.Markdown("# SodAI Drinks - Predictor de Compras")
    gr.Markdown("### Predicción automática para la próxima semana")
    
    with gr.Tabs():
        with gr.Tab("Predicción Individual"):
            gr.Markdown("""
            ### ℹPredicción para un cliente específico
            Predice si un cliente específico comprará un producto **en la próxima semana**.
            
            **Instrucciones:**
            1. Ingrese el ID del cliente
            2. Ingrese el ID del producto
            3. Presione "Predecir"
            """)
            
            with gr.Row():
                customer_input = gr.Number(label="ID del Cliente", value=10705, precision=0)
                product_input_ind = gr.Number(label="ID del Producto", value=1, precision=0)
            
            predict_ind_button = gr.Button("Predecir", variant="primary", size="lg")
            output_ind = gr.Markdown()
            
            predict_ind_button.click(
                fn=predict_individual,
                inputs=[customer_input, product_input_ind],
                outputs=output_ind
            )
        
        with gr.Tab("Predicción Agregada"):
            gr.Markdown("""
            ### Predicción para todos los clientes
            Calcula cuántos clientes comprarán un producto específico **en la próxima semana**.
            
            **Instrucciones:**
            1. Ingrese el ID del producto
            2. Presione "Calcular"
            
            **Nota:** Este cálculo puede tardar algunos segundos.
            """)
            
            product_input_agg = gr.Number(label="ID del Producto", value=1, precision=0)
            
            predict_agg_button = gr.Button("Calcular", variant="primary", size="lg")
            output_agg = gr.Markdown()
            
            predict_agg_button.click(
                fn=predict_aggregate,
                inputs=[product_input_agg],
                outputs=output_agg
            )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
