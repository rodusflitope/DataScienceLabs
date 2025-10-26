import gradio as gr
import requests
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://backend:8000")

def predict_purchase(customer_id, product_id, week):
    try:
        response = requests.post(
            f"{BACKEND_URL}/predict",
            json={
                "customer_id": int(customer_id),
                "product_id": int(product_id),
                "week": int(week)
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            prediction = "SI comprará" if result["prediction"] == 1 else "NO comprará"
            probability = result["prediction_proba"] * 100
            
            return f"""
## Resultado

**{prediction}**

**Probabilidad:** {probability:.1f}%

---
Cliente: {result['customer_id']} | Producto: {result['product_id']} | Semana: {result['week']}
            """
        else:
            error = response.json().get('detail', 'Error desconocido')
            return f"Error: {error}"
    
    except Exception as e:
        return f"Error de conexión: {str(e)}"

with gr.Blocks(title="SodAI Drinks") as app:
    gr.Markdown("# SodAI Drinks - Predictor de Compras")
    
    gr.Markdown("""
    ### Instrucciones
    1. Ingrese el ID del cliente
    2. Ingrese el ID del producto  
    3. Ingrese el número de semana
    4. Presione "Predecir"
    """)
    
    with gr.Row():
        customer_input = gr.Number(label="ID del Cliente", value=1, precision=0)
        product_input = gr.Number(label="ID del Producto", value=1, precision=0)
        week_input = gr.Number(label="Semana", value=1, precision=0)
    
    predict_button = gr.Button("Predecir", variant="primary", size="lg")
    output = gr.Markdown()
    
    predict_button.click(
        fn=predict_purchase,
        inputs=[customer_input, product_input, week_input],
        outputs=output
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
