# Proyección de Matrícula Escolar y Tasa de Escolarización
Panel interactivo en Streamlit para estimar, visualizar y descargar proyecciones de matrícula escolar mensual y anual, comparadas con la proyección demográfica por comuna, grado y establecimiento (RBD), incluyendo el cálculo de tasa de escolarización.

###Características
Proyección de matrícula mensual por comuna, RBD y grado, hasta 2027 (resto 2025, 2026, 2027).

Totales anuales automáticos por combinación.

Cálculo de tasa de escolarización al cruzar con proyección poblacional por edad y comuna.

Visualización interactiva: gráficos de líneas y barras (matrícula y tasa).

Descarga de Excel por comuna, por RBD y global.

Descarga de informes Word masivos por cada combinación seleccionada, con gráficos y tablas.

Metodología clara y documentada.


###Metodología
Proyección de matrícula:

Se utiliza Prophet para modelar y proyectar la matrícula mensual por cada combinación de comuna, RBD y grado.

Cruce demográfico:

Cada grado se mapea a una edad referencial.

Se compara la matrícula proyectada con la población proyectada por comuna y edad.

Se calcula la tasa de escolarización: matrícula proyectada / población proyectada.

Visualización y exportación:

El panel permite filtrar, visualizar, descargar Excel y generar informes Word para reportes.
