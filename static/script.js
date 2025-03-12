let inventarioData = [];
let ventasData = [];
const itemsPerPage = 5; // Número de elementos por página
let isDataLoaded = false; // Flag para controlar la carga de datos

function actualizarDatos() {
    $.getJSON('/actualizar_datos', function(data) {
        inventarioData = data.inventario;
        ventasData = data.ventas;
        mostrarInventario(1); // Mostrar la primera página del inventario
        mostrarVentas(1); // Mostrar la primera página de ventas
        isDataLoaded = true; // Marcar que los datos han sido cargados
        clearInterval(dataUpdateInterval); // Detener la actualización automática
    });
}

let dataUpdateInterval = setInterval(actualizarDatos, 5000); // Actualizar datos cada 5 segundos

function mostrarInventario(page) {
    const start = (page - 1) * itemsPerPage;
    const end = start + itemsPerPage;
    const paginatedData = inventarioData.slice(start, end);

    let inventarioHtml = '';
    paginatedData.forEach(function(producto) {
        inventarioHtml += `<tr>
            <td>${producto.id_producto}</td>
            <td>${producto.nombre}</td>
            <td>${producto.precio}</td>
            <td>${producto.cantidad}</td>
            <td>${producto.categoria}</td>
            <td>${producto.ultima_compra ? new Date(producto.ultima_compra).toLocaleString() : 'Nunca'}</td>
            <td><button onclick="mostrarHistorialVentas(${producto.id_producto})">Ver Historial</button></td>
        </tr>`;
    });
    $('#tabla-inventario tbody').html(inventarioHtml);
    mostrarPaginacion(page, inventarioData.length, '#pagination-inventario');
}

function mostrarVentas(page) {
    const start = (page - 1) * itemsPerPage;
    const end = start + itemsPerPage;
    const paginatedData = ventasData.slice(start, end);

    let ventasHtml = '';
    paginatedData.forEach(function(venta) {
        ventasHtml += `<tr>
            <td>${venta.id_venta}</td>
            <td>${new Date(venta.fecha).toLocaleString()}</td>
            <td>${venta.total}</td>
            <td>${venta.num_productos}</td>
            <td><button onclick="mostrarDetallesVenta(${venta.id_venta})">Detalles</button></td>
        </tr>`;
    });
    $('#tabla-ventas tbody').html(ventasHtml);
    mostrarPaginacion(page, ventasData.length, '#pagination-ventas');
}

function mostrarPaginacion(currentPage, totalItems, paginationSelector) {
    const totalPages = Math.ceil(totalItems / itemsPerPage);
    let paginationHtml = '';

    for (let i = 1; i <= totalPages; i++) {
        paginationHtml += `<button onclick="changePage(${i}, '${paginationSelector}')">${i}</button>`;
    }

    $(paginationSelector).html(paginationHtml);
}

function changePage(page, paginationSelector) {
    if (paginationSelector === '#pagination-inventario') {
        mostrarInventario(page);
    } else {
        mostrarVentas(page);
    }
}

function mostrarDetallesVenta(id_venta) {
    $.getJSON(`/detalles_venta/${id_venta}`, function(detalles) {
        let detallesHtml = `<h3>Detalles de la Venta ${id_venta}</h3>`;
        detallesHtml += `<p>Fecha: ${new Date(detalles.fecha).toLocaleString()}</p>`;
        detallesHtml += `<p>Total: $${detalles.total.toFixed(2)}</p>`;
        detallesHtml += `<h4>Productos Vendidos:</h4><ul>`;
        detalles.productos.forEach(function(producto) {
            detallesHtml += `<li>${producto.nombre} - Cantidad: ${producto.cantidad} - Precio Unitario: $${producto.precio_unitario.toFixed(2)}</li>`;
        });
        detallesHtml += `</ul>`;
        $('#detalles-venta').html(detallesHtml).show();
    });
}

function mostrarHistorialVentas(id_producto) {
    $.getJSON(`/historial_ventas/${id_producto}`, function(historial) {
        let historialHtml = `<h3>Historial de Ventas para Producto ${id_producto}</h3>`;
        historialHtml += `<ul>`;
        historial.forEach(function(venta) {
            historialHtml += `<li>Fecha: ${new Date(venta.fecha).toLocaleString()} - Cantidad: ${venta.cantidad}</li>`;
        });
        historialHtml += `</ul>`;
        $('#detalles-venta').html(historialHtml).show();
    });
}


function obtenerClasificacionProductos() {
    $.getJSON('/clasificacion_productos', function(data) {
        let html = '<h3>Clasificación de Productos</h3><table>';
        html += '<tr><th>ID</th><th>Nombre</th><th>Categoría</th><th>Cluster</th></tr>';
        data.forEach(producto => {
            html += `<tr>
                        <td>${producto.id_producto}</td>
                        <td>${producto.nombre}</td>
                        <td>${producto.categoria}</td>
                        <td>${producto.cluster}</td>
                    </tr>`;
        });
        html += '</table>';
        $('#resultado-tensorflow').html(html);
    }).fail(function() {
        $('#resultado-tensorflow').html("<p>Error al obtener la clasificación.</p>");
    });
}


function obtenerPrediccionVentas() {
    let id_producto = document.getElementById("producto-id").value;
    if (!id_producto) {
        alert("Por favor, ingresa un ID de producto.");
        return;
    }

    $.getJSON(`/prediccion_ventas/${id_producto}`, function(data) {
        if (data.predicciones) {
            let html = `<h3>Predicción de Ventas - ${data.producto}</h3><ul>`;
            Object.keys(data.predicciones).forEach(fecha => {
                html += `<li>${fecha}: ${data.predicciones[fecha]} unidades</li>`;
            });
            html += '</ul>';
            $('#resultado-pytorch').html(html);
        } else {
            $('#resultado-pytorch').html("<p>No hay predicciones disponibles.</p>");
        }
    }).fail(function() {
        $('#resultado-pytorch').html("<p>Error al obtener la predicción.</p>");
    });
}

function obtenerReabastecimiento() {
    $.getJSON('/reabastecimiento', function(data) {
        let html = '<h2>Recomendaciones de Reabastecimiento</h2><ul>';
        data.forEach(rec => {
            html += `<li>${rec.nombre} - Urgencia: ${rec.urgencia} - Sugerido: ${rec.cantidad_sugerida}</li>`;
        });
        html += '</ul>';
        $('#contenido-adicional').html(html);
    });
}

function obtenerProductosBajoRendimiento() {
    $.getJSON('/productos_bajo_rendimiento', function(data) {
        let html = '<h2>Productos de Bajo Rendimiento</h2><table>';
        html += '<tr><th>Nombre</th><th>Frecuencia Ventas</th><th>Días sin Venta</th></tr>';
        data.forEach(prod => {
            html += `<tr>
                        <td>${prod.nombre}</td>
                        <td>${prod.frecuencia_ventas}</td>
                        <td>${prod.dias_sin_venta}</td>
                    </tr>`;
        });
        html += '</table>';
        $('#contenido-adicional').html(html);
    });
}

function obtenerPatronesVentas() {
    $.getJSON('/patrones_ventas', function(data) {
        let html = '<h2>Patrones de Ventas</h2><h3>Por Día de la Semana</h3><ul>';
        data.ventas_por_dia.forEach(p => {
            html += `<li>${p.Día}: ${p['Total Ventas']} ventas</li>`;
        });
        html += '</ul><h3>Por Hora del Día</h3><ul>';
        data.ventas_por_hora.forEach(p => {
            html += `<li>${p.Hora}: ${p['Total Ventas']} ventas</li>`;
        });
        html += '</ul>';
        $('#contenido-adicional').html(html);
    });
}


function actualizarLogs() {
    $.getJSON('/obtener_logs', function(data) {
        let logsHtml = "<h2>Registros del Sistema</h2><ul>";
        data.logs.forEach(function(log) {
            logsHtml += `<li>${log}</li>`;
        });
        logsHtml += "</ul>";

        $('#logs-sistema').html(logsHtml);
    });
}

// Llamar a la función cada 5 segundos
setInterval(actualizarLogs, 5000);
