# Sistema Comercial para Tienda de Abarrotes con Inteligencia Artificial
# Integración de PyTorch y TensorFlow para predictores y clasificadores

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Librerías de IA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

# Clase base para el sistema de tienda
class SistemaAbarrotes:
    def __init__(self, nombre_tienda):
        self.nombre_tienda = nombre_tienda
        self.inventario = {}
        self.ventas = []
        self.categorias_productos = set()
        self.predictor_ventas = None
        self.clasificador_productos = None
        print(f"Sistema inicializado para: {self.nombre_tienda}")
    
    def agregar_producto(self, id_producto, nombre, precio, cantidad, categoria):
        self.inventario[id_producto] = {
            'nombre': nombre,
            'precio': precio,
            'cantidad': cantidad,
            'categoria': categoria,
            'historial_ventas': [],
            'ultima_compra': None
        }
        self.categorias_productos.add(categoria)
        print(f"Producto agregado: {nombre}")
    
    def registrar_venta(self, id_venta, productos, fecha=None):
        if fecha is None:
            fecha = datetime.now()
        
        total = 0
        productos_vendidos = []
        
        for id_prod, cantidad in productos:
            if id_prod in self.inventario and self.inventario[id_prod]['cantidad'] >= cantidad:
                precio_unitario = self.inventario[id_prod]['precio']
                subtotal = precio_unitario * cantidad
                total += subtotal
                
                # Actualizar inventario
                self.inventario[id_prod]['cantidad'] -= cantidad
                self.inventario[id_prod]['historial_ventas'].append({
                    'fecha': fecha,
                    'cantidad': cantidad
                })
                self.inventario[id_prod]['ultima_compra'] = fecha
                
                productos_vendidos.append({
                    'id_producto': id_prod,
                    'nombre': self.inventario[id_prod]['nombre'],
                    'cantidad': cantidad,
                    'precio_unitario': precio_unitario,
                    'subtotal': subtotal
                })
            else:
                print(f"Error: Producto {id_prod} no disponible en la cantidad solicitada")
                return None
        
        venta = {
            'id_venta': id_venta,
            'fecha': fecha,
            'productos': productos_vendidos,
            'total': total
        }
        
        self.ventas.append(venta)
        
        print(f"Venta {id_venta} registrada. Total: ${total:.2f}")
        return venta
    
    def generar_reporte_inventario(self):
        df = pd.DataFrame([
            {
                'id_producto': id_prod,
                'nombre': info['nombre'],
                'categoria': info['categoria'],
                'precio': info['precio'],
                'cantidad': info['cantidad'],
                'valor_total': info['precio'] * info['cantidad'],
                'dias_sin_venta': (datetime.now() - info['ultima_compra']).days if info['ultima_compra'] else None
            }
            for id_prod, info in self.inventario.items()
        ])
        
        return df
    
    def generar_reporte_ventas(self, fecha_inicio=None, fecha_fin=None):
        if fecha_inicio is None:
            fecha_inicio = datetime.min
        if fecha_fin is None:
            fecha_fin = datetime.max
        
        ventas_filtradas = [
            venta for venta in self.ventas 
            if fecha_inicio <= venta['fecha'] <= fecha_fin
        ]
        
        df = pd.DataFrame([
            {
                'id_venta': venta['id_venta'],
                'fecha': venta['fecha'],
                'total': venta['total'],
                'num_productos': len(venta['productos'])
            }
            for venta in ventas_filtradas
        ])
        
        return df
    
    def generar_reporte_ventas_por_categoria(self, fecha_inicio=None, fecha_fin=None):
        if fecha_inicio is None:
            fecha_inicio = datetime.min
        if fecha_fin is None:
            fecha_fin = datetime.max
        
        ventas_categoria = {cat: 0 for cat in self.categorias_productos}
        cantidad_categoria = {cat: 0 for cat in self.categorias_productos}
        
        for venta in self.ventas:
            if fecha_inicio <= venta['fecha'] <= fecha_fin:
                for producto in venta['productos']:
                    id_prod = producto['id_producto']
                    if id_prod in self.inventario:
                        categoria = self.inventario[id_prod]['categoria']
                        ventas_categoria[categoria] += producto['subtotal']
                        cantidad_categoria[categoria] += producto['cantidad']
        
        df = pd.DataFrame([
            {
                'categoria': cat,
                'ventas_total': ventas_categoria[cat],
                'cantidad_total': cantidad_categoria[cat]
            }
            for cat in self.categorias_productos
        ])
        
        return df
    
    def generar_datos_entrenamiento(self):
        # Crear dataset para entrenamiento de modelos IA
        datos_productos = []
        
        for id_prod, info in self.inventario.items():
            historial = info['historial_ventas']
            if historial:
                # Agrupar ventas por día
                ventas_diarias = {}
                for venta in historial:
                    fecha_str = venta['fecha'].strftime('%Y-%m-%d')
                    if fecha_str in ventas_diarias:
                        ventas_diarias[fecha_str] += venta['cantidad']
                    else:
                        ventas_diarias[fecha_str] = venta['cantidad']
                
                # Crear series temporales
                for fecha_str, cantidad in ventas_diarias.items():
                    datos_productos.append({
                        'id_producto': id_prod,
                        'nombre': info['nombre'],
                        'categoria': info['categoria'],
                        'precio': info['precio'],
                        'fecha': datetime.strptime(fecha_str, '%Y-%m-%d'),
                        'cantidad_vendida': cantidad
                    })
        
        return pd.DataFrame(datos_productos)
    
    def analizar_productos_relacionados(self):
        # Análisis de productos que se compran juntos frecuentemente
        relaciones = {}
        
        for venta in self.ventas:
            productos = [p['id_producto'] for p in venta['productos']]
            
            for i in range(len(productos)):
                for j in range(i+1, len(productos)):
                    par = tuple(sorted([productos[i], productos[j]]))
                    if par in relaciones:
                        relaciones[par] += 1
                    else:
                        relaciones[par] = 1
        
        # Convertir a DataFrame
        df_relaciones = pd.DataFrame([
            {
                'producto1': self.inventario[par[0]]['nombre'],
                'producto2': self.inventario[par[1]]['nombre'],
                'frecuencia': freq
            }
            for par, freq in relaciones.items()
            if freq > 1  # Filtrar relaciones poco frecuentes
        ]).sort_values('frecuencia', ascending=False)
        
        return df_relaciones

# Clase para el manejo de datos
class ProcesadorDatos:
    def __init__(self):
        self.scaler_x = StandardScaler()
        self.scaler_y = StandardScaler()
        self.encoders = {}
    
    def preparar_datos_series_tiempo(self, df, ventana=7, horizonte=1):
        # Ordenar por fecha
        df_ordenado = df.sort_values('fecha')
        
        # Crear características de series temporales
        X, y = [], []
        
        productos_unicos = df_ordenado['id_producto'].unique()
        
        for id_prod in productos_unicos:
            df_prod = df_ordenado[df_ordenado['id_producto'] == id_prod]
            
            # Verificar que haya suficientes datos
            if len(df_prod) < ventana + horizonte:
                continue
                
            serie = df_prod['cantidad_vendida'].values
            
            for i in range(len(serie) - ventana - horizonte + 1):
                X.append(serie[i:i+ventana])
                y.append(serie[i+ventana:i+ventana+horizonte])
        
        return np.array(X), np.array(y)
    
    def preparar_datos_productos(self, sistema):
        # Extraer características de productos para clasificación
        datos_productos = []
        
        for id_prod, info in sistema.inventario.items():
            # Calcular frecuencia de venta
            freq = len(info['historial_ventas'])
            
            # Calcular promedio de ventas diarias
            if freq > 0:
                primera_venta = min(venta['fecha'] for venta in info['historial_ventas'])
                ultima_venta = max(venta['fecha'] for venta in info['historial_ventas'])
                dias = max(1, (ultima_venta - primera_venta).days + 1)
                total_vendido = sum(venta['cantidad'] for venta in info['historial_ventas'])
                prom_diario = total_vendido / dias
            else:
                prom_diario = 0
            
            datos_productos.append({
                'id_producto': id_prod,
                'nombre': info['nombre'],
                'categoria': info['categoria'],
                'precio': info['precio'],
                'cantidad_actual': info['cantidad'],
                'frecuencia_ventas': freq,
                'promedio_diario': prom_diario,
                'valor_inventario': info['precio'] * info['cantidad']
            })
        
        return pd.DataFrame(datos_productos)
    
    def preparar_para_modelo(self, df, columnas_cat=[], columnas_num=[]):
        # Codificar variables categóricas
        X = df.copy()
        
        for col in columnas_cat:
            if col in X.columns:
                if col not in self.encoders:
                    self.encoders[col] = LabelEncoder()
                    self.encoders[col].fit(X[col].astype(str))
                X[col] = self.encoders[col].transform(X[col].astype(str))
        
        # Escalar variables numéricas
        if columnas_num:
            X_num = X[columnas_num].values
            X_num = self.scaler_x.fit_transform(X_num)
            for i, col in enumerate(columnas_num):
                X[col] = X_num[:, i]
        
        return X

# Modelo PyTorch para predicción de ventas
class PredictorVentasPyTorch(nn.Module):
    def __init__(self, input_size=7, hidden_size=64, output_size=1):
        super(PredictorVentasPyTorch, self).__init__()
        self.lstm = nn.LSTM(1, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Reshape input: [batch, seq_len] -> [batch, seq_len, 1]
        x = x.view(x.size(0), x.size(1), 1)
        out, _ = self.lstm(x)
        # Tomamos solo la última salida de la secuencia
        out = self.linear(out[:, -1, :])
        return out

# Dataset personalizado para PyTorch
class VentasDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Modelo TensorFlow para clasificación de productos
def crear_modelo_tf_clasificador(input_size, num_clases):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(input_size,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(num_clases, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Interfaz principal que une todo el sistema
class SistemaComercialInteligente:
    def __init__(self, nombre_tienda):
        self.sistema = SistemaAbarrotes(nombre_tienda)
        self.procesador = ProcesadorDatos()
        self.modelo_prediccion = None
        self.modelo_clasificacion = None
        
    def cargar_datos_muestra(self):
        # Productos de muestra
        categorias = ['Lácteos', 'Carnes', 'Frutas', 'Verduras', 'Abarrotes', 'Bebidas', 'Limpieza']
        
        for i in range(1, 31):
            categoria = categorias[i % len(categorias)]
            precio = round(np.random.uniform(10, 200), 2)
            cantidad = np.random.randint(10, 100)
            self.sistema.agregar_producto(
                id_producto=i,
                nombre=f"Producto {i}",
                precio=precio,
                cantidad=cantidad,
                categoria=categoria
            )
        
        # Generar ventas simuladas
        fechas = [datetime.now() - timedelta(days=i) for i in range(60, 0, -1)]
        
        for i in range(1, 201):
            fecha = fechas[np.random.randint(0, len(fechas))]
            num_productos = np.random.randint(1, 6)
            
            productos = []
            for _ in range(num_productos):
                id_prod = np.random.randint(1, 31)
                cantidad = np.random.randint(1, 5)
                productos.append((id_prod, cantidad))
            
            self.sistema.registrar_venta(
                id_venta=i,
                productos=productos,
                fecha=fecha
            )
    
    def entrenar_predictor_ventas(self):
        # Obtener datos de entrenamiento
        df_ventas = self.sistema.generar_datos_entrenamiento()
        
        if len(df_ventas) < 10:
            print("Datos insuficientes para entrenar el predictor de ventas")
            return False
        
        # Preparar datos para series temporales
        X, y = self.procesador.preparar_datos_series_tiempo(df_ventas)
        
        if len(X) < 10:
            print("Datos insuficientes después de crear ventanas temporales")
            return False
            
        # Dividir en conjuntos de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Crear y entrenar modelo PyTorch
        model = PredictorVentasPyTorch(input_size=X.shape[1])
        
        # Parámetros de entrenamiento
        batch_size = 16
        epochs = 50
        learning_rate = 0.001
        
        # Crear datasets y dataloaders
        train_dataset = VentasDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Definir criterio y optimizador
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        # Entrenamiento
        model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Época {epoch+1}/{epochs}, Pérdida: {running_loss/len(train_loader):.4f}')
        
        # Guardar modelo
        self.modelo_prediccion = model
        print("Modelo de predicción de ventas entrenado exitosamente")
        return True
    
    def clasificar_productos(self, num_clusters=3):
        """
        Clasifica productos según su comportamiento de ventas usando TensorFlow
        """
        # Obtener datos de productos
        df_productos = self.procesador.preparar_datos_productos(self.sistema)
        
        if len(df_productos) < 10:
            print("Datos insuficientes para clasificar productos")
            return False
        
        # Características para clustering
        features = [
            'precio', 'frecuencia_ventas', 'promedio_diario', 'cantidad_actual'
        ]
        
        # Preparar datos
        X = df_productos[features].values
        
        # Normalizar
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Clustering con K-means
        kmeans = KMeans(n_clusters=num_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Agregar cluster a dataframe
        df_productos['cluster'] = clusters
        
        # Analizar los clusters
        cluster_info = {}
        for i in range(num_clusters):
            cluster_data = df_productos[df_productos['cluster'] == i]
            
            # Determinar características del cluster
            precio_medio = cluster_data['precio'].mean()
            frec_media = cluster_data['frecuencia_ventas'].mean()
            prom_diario = cluster_data['promedio_diario'].mean()
            
            # Asignar etiqueta descriptiva
            if precio_medio > df_productos['precio'].mean() + df_productos['precio'].std():
                precio_desc = "Alto Precio"
            elif precio_medio < df_productos['precio'].mean() - df_productos['precio'].std():
                precio_desc = "Bajo Precio"
            else:
                precio_desc = "Precio Medio"
                
            if frec_media > df_productos['frecuencia_ventas'].mean() + df_productos['frecuencia_ventas'].std():
                frec_desc = "Alta Rotación"
            elif frec_media < df_productos['frecuencia_ventas'].mean() - df_productos['frecuencia_ventas'].std():
                frec_desc = "Baja Rotación"
            else:
                frec_desc = "Rotación Media"
            
            etiqueta = f"{precio_desc}, {frec_desc}"
            
            cluster_info[i] = {
                'etiqueta': etiqueta,
                'precio_medio': precio_medio,
                'frecuencia_media': frec_media,
                'promedio_diario': prom_diario,
                'conteo': len(cluster_data)
            }
            
            print(f"Cluster {i} ({etiqueta}): {len(cluster_data)} productos")
            
        # Crear modelo de clasificación con TensorFlow
        # Entrenar en base a las etiquetas de clustering
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, clusters, test_size=0.2, random_state=42
        )
        
        # Crear y entrenar modelo TensorFlow
        model = crear_modelo_tf_clasificador(input_size=len(features), num_clases=num_clusters)
        
        # Entrenar modelo
        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=16,
            validation_data=(X_test, y_test),
            verbose=0
        )
        
        # Evaluar modelo
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"Precisión del clasificador: {test_acc:.4f}")
        
        # Guardar modelo y datos relacionados
        self.modelo_clasificacion = model
        self.scaler_clasificacion = scaler
        self.features_clasificacion = features
        self.cluster_info = cluster_info
        self.df_productos_clasificados = df_productos
        
        return df_productos
    
    def predecir_ventas_producto(self, id_producto, dias_historico=7, dias_futuro=7):
        if self.modelo_prediccion is None:
            print("El modelo de predicción no ha sido entrenado")
            return None
        
        if id_producto not in self.sistema.inventario:
            print(f"Producto {id_producto} no encontrado")
            return None
        
        # Obtener historial de ventas
        historial = self.sistema.inventario[id_producto]['historial_ventas']
        
        if len(historial) < dias_historico:
            print(f"Datos insuficientes. Se necesitan al menos {dias_historico} días de historial")
            return None
        
        # Agrupar ventas por día
        ventas_diarias = {}
        for venta in historial:
            fecha_str = venta['fecha'].strftime('%Y-%m-%d')
            if fecha_str in ventas_diarias:
                ventas_diarias[fecha_str] += venta['cantidad']
            else:
                ventas_diarias[fecha_str] = venta['cantidad']
        
        # Ordenar fechas
        fechas_ordenadas = sorted(ventas_diarias.keys())
        
        # Tomar los últimos días_historico días
        ultimos_dias = fechas_ordenadas[-dias_historico:]
        ultimas_ventas = [ventas_diarias[fecha] for fecha in ultimos_dias]
        
        # Preparar datos para predicción
        X = torch.tensor([ultimas_ventas], dtype=torch.float32)
        
        # Realizar predicciones para los días futuros
        self.modelo_prediccion.eval()
        with torch.no_grad():
            predicciones = []
            input_seq = X.clone()
            
            for _ in range(dias_futuro):
                pred = self.modelo_prediccion(input_seq)
                predicciones.append(pred.item())
                
                # Actualizar secuencia de entrada para la siguiente predicción
                input_seq = torch.cat([input_seq[:, 1:], pred.reshape(1, 1)], dim=1)
        
        # Crear fechas futuras
        ultima_fecha = datetime.strptime(ultimos_dias[-1], '%Y-%m-%d')
        fechas_futuras = [(ultima_fecha + timedelta(days=i+1)).strftime('%Y-%m-%d') 
                           for i in range(dias_futuro)]
        
        # Presentar resultados
        resultado = {
            'producto': self.sistema.inventario[id_producto]['nombre'],
            'historial': dict(zip(ultimos_dias, ultimas_ventas)),
            'predicciones': dict(zip(fechas_futuras, [max(0, round(p)) for p in predicciones]))
        }
        
        return resultado
    
    def predecir_agotamiento_stock(self, id_producto):
        """
        Predice cuándo se agotará el stock de un producto
        """
        prediccion = self.predecir_ventas_producto(id_producto, dias_futuro=30)
        if not prediccion:
            return None
        
        stock_actual = self.sistema.inventario[id_producto]['cantidad']
        consumo_acumulado = 0
        
        for fecha, cantidad in prediccion['predicciones'].items():
            consumo_acumulado += cantidad
            if consumo_acumulado >= stock_actual:
                return {
                    'producto': prediccion['producto'],
                    'stock_actual': stock_actual,
                    'fecha_agotamiento': fecha,
                    'dias_hasta_agotamiento': (datetime.strptime(fecha, '%Y-%m-%d') - datetime.now()).days
                }
        
        return {
            'producto': prediccion['producto'],
            'stock_actual': stock_actual,
            'fecha_agotamiento': 'Más de 30 días',
            'dias_hasta_agotamiento': '>30'
        }
    
    def recomendar_reabastecimiento(self):
        """
        Genera recomendaciones de reabastecimiento basadas en las predicciones
        """
        recomendaciones = []
        
        for id_producto in self.sistema.inventario:
            prediccion_agotamiento = self.predecir_agotamiento_stock(id_producto)
            if prediccion_agotamiento and prediccion_agotamiento['dias_hasta_agotamiento'] != '>30':
                if prediccion_agotamiento['dias_hasta_agotamiento'] <= 7:
                    urgencia = 'Alta'
                elif prediccion_agotamiento['dias_hasta_agotamiento'] <= 15:
                    urgencia = 'Media'
                else:
                    urgencia = 'Baja'
                
                # Calcular cantidad sugerida
                ventas_recientes = sum(v['cantidad'] for v in self.sistema.inventario[id_producto]['historial_ventas'][-30:] if v)
                dias_ventas = min(30, len(self.sistema.inventario[id_producto]['historial_ventas']))
                venta_diaria_promedio = ventas_recientes / max(1, dias_ventas)
                
                # Sugerir stock para 30 días
                cantidad_sugerida = max(10, round(venta_diaria_promedio * 30))
                
                recomendaciones.append({
                    'id_producto': id_producto,
                    'nombre': self.sistema.inventario[id_producto]['nombre'],
                    'categoria': self.sistema.inventario[id_producto]['categoria'],
                    'stock_actual': self.sistema.inventario[id_producto]['cantidad'],
                    'dias_hasta_agotamiento': prediccion_agotamiento['dias_hasta_agotamiento'],
                    'urgencia': urgencia,
                    'cantidad_sugerida': cantidad_sugerida
                })
        
        # Ordenar por urgencia
        return sorted(recomendaciones, key=lambda x: (
            0 if x['urgencia'] == 'Alta' else 1 if x['urgencia'] == 'Media' else 2,
            x['dias_hasta_agotamiento']
        ))
    
    def identificar_productos_bajo_rendimiento(self):
        """
        Identifica productos con bajo rendimiento de ventas
        """
        df_productos = self.procesador.preparar_datos_productos(self.sistema)
        
        # Calcular días desde última venta
        for i, row in df_productos.iterrows():
            id_prod = row['id_producto']
            ultima_venta = self.sistema.inventario[id_prod]['ultima_compra']
            if ultima_venta:
                df_productos.at[i, 'dias_sin_venta'] = (datetime.now() - ultima_venta).days
            else:
                df_productos.at[i, 'dias_sin_venta'] = float('inf')
        
        # Criterios de bajo rendimiento
        df_productos['bajo_rendimiento'] = (
            (df_productos['frecuencia_ventas'] < df_productos['frecuencia_ventas'].quantile(0.25)) & 
            (df_productos['dias_sin_venta'] > 30) & 
            (df_productos['valor_inventario'] > 0)
        )
        
        # Productos de bajo rendimiento
        bajo_rendimiento = df_productos[df_productos['bajo_rendimiento']]
        
        if len(bajo_rendimiento) == 0:
            return None
        
        return bajo_rendimiento.sort_values('frecuencia_ventas')
    
    def analizar_patrones_temporales(self):
        """
        Analiza patrones temporales de ventas por día de la semana y hora
        """
        # Convertir ventas a DataFrame
        ventas_data = []
        
        for venta in self.sistema.ventas:
            fecha = venta['fecha']
            dia_semana = fecha.strftime('%A')
            hora = fecha.hour
            
            ventas_data.append({
                'id_venta': venta['id_venta'],
                'fecha': fecha,
                'dia_semana': dia_semana,
                'hora': hora,
                'total': venta['total'],
                'num_productos': len(venta['productos'])
            })
        
        df_ventas = pd.DataFrame(ventas_data)
        
        # Análisis por día de la semana
        ventas_por_dia = df_ventas.groupby('dia_semana').agg({
            'id_venta': 'count',
            'total': 'sum',
            'num_productos': 'sum'
        }).reset_index()
        
        ventas_por_dia.columns = ['Día', 'Número Ventas', 'Total Ventas', 'Productos Vendidos']
        
        # Análisis por hora
        df_ventas['hora_grupo'] = df_ventas['hora'].apply(
            lambda x: f"{x}-{x+1}" if x < 23 else "23-24"
        )
        
        ventas_por_hora = df_ventas.groupby('hora_grupo').agg({
            'id_venta': 'count',
            'total': 'sum',
            'num_productos': 'sum'
        }).reset_index()
        
        ventas_por_hora.columns = ['Hora', 'Número Ventas', 'Total Ventas', 'Productos Vendidos']
        
        return {
            'ventas_por_dia': ventas_por_dia,
            'ventas_por_hora': ventas_por_hora
        }
    
    def visualizar_prediccion(self, prediccion):
        if not prediccion:
            return
        
        # Crear gráfica de predicción
        historial = prediccion['historial']
        predicciones = prediccion['predicciones']
    
        fechas_hist = list(historial.keys())
        valores_hist = list(historial.values())
        
        fechas_pred = list(predicciones.keys())
        valores_pred = list(predicciones.values())
        
        plt.figure(figsize=(12, 6))
        plt.plot(fechas_hist, valores_hist, 'b-o', label='Historial')
        plt.plot(fechas_pred, valores_pred, 'r-o', label='Predicción')
        plt.axvline(x=len(fechas_hist)-0.5, color='green', linestyle='--', label='Hoy')
        
        plt.title(f'Predicción de ventas: {prediccion["producto"]}')
        plt.xlabel('Fecha')
        plt.ylabel('Unidades')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.show()