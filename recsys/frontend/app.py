import gradio as gr
import requests
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8001")

def get_recommendations(customer_id):
    if not customer_id:
        return "Please enter a valid customer ID."
    
    try:
        customer_id = int(customer_id)
        response = requests.post(
            f"{BACKEND_URL}/recommend",
            json={"customer_id": customer_id}
        )
        
        if response.status_code == 200:
            data = response.json()
            
            result = f"## Recommendations for Customer {data['customer_id']} (Week {data['week']})\n\n"
            
            for i, rec in enumerate(data['recommendations'], 1):
                result += f"### {i}. {rec['product_name']}\n"
                result += f"- Brand: {rec['brand']}\n"
                result += f"- Category: {rec['category']}\n"
                result += f"- Sub-category: {rec['sub_category']}\n"
                result += f"- Purchase Probability: {rec['prediction_proba']:.2%}\n\n"
            
            return result
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            return f"Error: {error_detail}"
    except ValueError:
        return "Error: Customer ID must be a number."
    except requests.exceptions.RequestException as e:
        return f"Error connecting to backend: {str(e)}"

with gr.Blocks(title="SodAI Drinks - Recommendation System") as app:
    gr.Markdown("# SodAI Drinks - Product Recommendation System")
    gr.Markdown("Get the top 5 product recommendations for any customer based on purchase probability.")
    
    with gr.Row():
        customer_input = gr.Number(label="Customer ID", precision=0)
        recommend_btn = gr.Button("Get Recommendations")
    
    output = gr.Markdown(label="Recommendations")
    
    recommend_btn.click(
        fn=get_recommendations,
        inputs=[customer_input],
        outputs=[output]
    )
    
    gr.Markdown("""
    ### Instructions
    1. Enter a customer ID (e.g., 1, 2, 3...)
    2. Click "Get Recommendations"
    3. The system will show the top 5 products with highest purchase probability for the next week
    
    Note: Recommendations are automatically calculated for the next week based on the latest data.
    """)

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7861)
