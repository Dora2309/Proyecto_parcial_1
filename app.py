from flask import Flask, render_template, jsonify
import threading
import time
import pandas as pd
import os
from sistema_tienda import SistemaComercialInteligente


# Configurar Flask correctamente
app = Flask(__name__)
sistema = SistemaComercialInteligente("Mi Tienda de Abarrotes")

app = Flask(__name__)
sistema = SistemaComercialInteligente("Mi Tienda de Abarrotes")

# Cargar datos de muestra y entrenar el modelo al iniciar
sistema.cargar_datos_muestra()
sistema.entrenar_predictor_ventas()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/actualizar_datos')
def actualizar_datos():
    # Generar reportes o datos que deseas mostrar
    reporte_inventario = sistema.sistema.generar_reporte_inventario().to_dict(orient='records')
    reporte_ventas = sistema.sistema.generar_reporte_ventas().to_dict(orient='records')
    
    return jsonify({
        'inventario': reporte_inventario,
        'ventas': reporte_ventas
    })


@app.route('/detalles_venta/<int:id_venta>')
def detalles_venta(id_venta):
    for venta in sistema.sistema.ventas:
        if venta['id_venta'] == id_venta:
            return jsonify(venta)
    return jsonify({}), 404

@app.route('/historial_ventas/<int:id_producto>')
def historial_ventas(id_producto):
    # Buscar el producto en el inventario
    for id_prod, producto in sistema.sistema.inventario.items():
        if id_prod == id_producto:
            return jsonify(producto['historial_ventas'])
    return jsonify([]), 404

@app.route('/clasificacion_productos')
def clasificacion_productos():
    resultado = sistema.clasificar_productos()
    if isinstance(resultado, pd.DataFrame):
        return jsonify(resultado.to_dict(orient='records'))
    return jsonify([])


@app.route('/prediccion_ventas/<int:id_producto>')
def prediccion_ventas(id_producto):
    resultado = sistema.predecir_ventas_producto(id_producto, dias_futuro=14)
    if resultado:
        return jsonify(resultado)
    return jsonify({}), 404

@app.route('/reabastecimiento')
def reabastecimiento():
    resultado = sistema.recomendar_reabastecimiento()
    return jsonify(resultado)

@app.route('/productos_bajo_rendimiento')
def productos_bajo_rendimiento():
    resultado = sistema.identificar_productos_bajo_rendimiento()
    if resultado is not None:
        return jsonify(resultado.to_dict(orient='records'))
    return jsonify([])

@app.route('/patrones_ventas')
def patrones_ventas():
    resultado = sistema.analizar_patrones_temporales()
    return jsonify({
        'ventas_por_dia': resultado['ventas_por_dia'].to_dict(orient='records'),
        'ventas_por_hora': resultado['ventas_por_hora'].to_dict(orient='records')
    })

def run_flask():
    app.run(debug=True, use_reloader=False)

    
@app.route('/obtener_logs')
def obtener_logs():
    from ejecutar_sistema import get_logs  # Importamos la función de logs
    return get_logs()

if __name__ == '__main__':
    threading.Thread(target=run_flask).start()



    