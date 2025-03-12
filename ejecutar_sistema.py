from sistema_tienda import SistemaComercialInteligente
import matplotlib.pyplot as plt
import pandas as pd
import json

# Configurar visualización de pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Variable global para almacenar logs del sistema
logs_sistema = []

def log(mensaje):
    """ Agrega un mensaje al registro del sistema """
    logs_sistema.append(mensaje)
    print(mensaje)  # Mantiene la impresión en la consola


def get_logs():
   """ Devuelve los logs en formato JSON """
   return json.dumps({"logs": logs_sistema})


def main():
    # Crear instancia del sistema
    log("Inicializando Sistema Comercial Inteligente...")
    sistema = SistemaComercialInteligente("Mi Tienda de Abarrotes")
    
    # Cargar datos de muestra
    log("\nCargando datos de muestra...")
    sistema.cargar_datos_muestra()
    
    # Ver reporte básico de inventario
    log("\nGenerando reporte de inventario:")
    reporte_inventario = sistema.sistema.generar_reporte_inventario()
    log(reporte_inventario.head().to_string(index=False))
    
    # Ver reporte de ventas
    log("\nGenerando reporte de ventas:")
    reporte_ventas = sistema.sistema.generar_reporte_ventas()
    log(reporte_ventas.head().to_string(index=False))
    
    # Entrenar predictor de ventas
    log("\nEntrenando predictor de ventas...")
    sistema.entrenar_predictor_ventas()
    
    # Clasificar productos
    log("\nClasificando productos...")
    productos_clasificados = sistema.clasificar_productos()
    log("\nResumen de clasificación de productos:")
    log(productos_clasificados[['nombre', 'categoria', 'cluster']].head(10).to_string(index=False))
    
    
    # Predecir ventas para un producto específico
    id_producto = 5  # Producto de ejemplo
    log(f"\nPrediciendo ventas futuras para el producto {id_producto}...")
    prediccion = sistema.predecir_ventas_producto(id_producto, dias_futuro=14)
    if prediccion:
        log(f"Producto: {prediccion['producto']}")
        log("Historial de ventas:")
        for fecha, cantidad in prediccion['historial'].items():
            log(f"  {fecha}: {cantidad}")
        log("Predicciones:")
        for fecha, cantidad in prediccion['predicciones'].items():
            log(f"  {fecha}: {cantidad}")
    
    # Obtener recomendaciones de reabastecimiento
    log("\nGenerando recomendaciones de reabastecimiento...")
    recomendaciones = sistema.recomendar_reabastecimiento()
    if recomendaciones:
        log("Top 5 productos que necesitan reabastecimiento:")
        for i, rec in enumerate(recomendaciones[:5]):
            log(f"{i+1}. {rec['nombre']} - Stock actual: {rec['stock_actual']} - Urgencia: {rec['urgencia']} - Cantidad sugerida: {rec['cantidad_sugerida']}")
    
    # Identificar productos de bajo rendimiento
    log("\nIdentificando productos de bajo rendimiento...")
    bajo_rendimiento = sistema.identificar_productos_bajo_rendimiento()
    if bajo_rendimiento is not None:
        log("Productos con bajo rendimiento:")
        log(bajo_rendimiento[['nombre', 'frecuencia_ventas', 'dias_sin_venta']].head().to_string(index=False))

    # Analizar patrones temporales
    log("\nAnalizando patrones temporales de ventas...")
    patrones = sistema.analizar_patrones_temporales()
    log("Ventas por día de la semana:")
    log(patrones['ventas_por_dia'].to_string(index=False))
    log("\nVentas por hora del día:")
    log(patrones['ventas_por_hora'].head().to_string(index=False))
    
    # Analizar productos relacionados
    log("\nAnalizando productos frecuentemente comprados juntos...")
    productos_relacionados = sistema.sistema.analizar_productos_relacionados()
    if not productos_relacionados.empty:
        log(productos_relacionados.head(10).to_string(index=False))
    
    log("\nAnálisis del sistema completado.")

if __name__ == "__main__":
    main()

