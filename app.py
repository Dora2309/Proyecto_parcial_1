import gradio as gr
import pandas as pd
from sistema_tienda import SistemaComercialInteligente

# Cargar el sistema de tienda
sistema = SistemaComercialInteligente("Mi Tienda de Abarrotes")

# Función para mostrar el inventario
def mostrar_inventario():
    df = sistema.sistema.generar_reporte_inventario()
    return df

# Función para predecir ventas
def predecir_ventas(id_producto):
    resultado = sistema.predecir_ventas_producto(id_producto, dias_futuro=14)
    if resultado:
        return resultado
    return "No hay predicciones disponibles."

# Función para ver días sin venta
def dias_sin_venta(id_producto):
    if id_producto in sistema.sistema.inventario:
        producto = sistema.sistema.inventario[id_producto]
        ultima_compra = producto.get('ultima_compra')

        if ultima_compra:
            dias_sin_venta = (pd.Timestamp.now() - pd.Timestamp(ultima_compra)).days
        else:
            dias_sin_venta = "Nunca se ha vendido"

        return f"Última compra: {ultima_compra}, Días sin venta: {dias_sin_venta}"
    return "Producto no encontrado."

# Crear la interfaz en Gradio
with gr.Blocks() as interfaz:
    gr.Markdown("# 📦 Sistema Comercial Inteligente")
    
    with gr.Tab("Inventario"):
        btn_inventario = gr.Button("Mostrar Inventario")
        salida_inventario = gr.Dataframe()
        btn_inventario.click(mostrar_inventario, outputs=salida_inventario)
    
    with gr.Tab("Predicción de Ventas"):
        entrada_id = gr.Number(label="ID del Producto")
        btn_prediccion = gr.Button("Predecir Ventas")
        salida_prediccion = gr.Textbox()
        btn_prediccion.click(predecir_ventas, inputs=entrada_id, outputs=salida_prediccion)
    
    with gr.Tab("Días sin Venta"):
        entrada_id_dias = gr.Number(label="ID del Producto")
        btn_dias = gr.Button("Ver Días sin Venta")
        salida_dias = gr.Textbox()
        btn_dias.click(dias_sin_venta, inputs=entrada_id_dias, outputs=salida_dias)

# Ejecutar la interfaz
if __name__ == "__main__":
    interfaz.launch()
