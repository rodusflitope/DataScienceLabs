# Conclusiones del Proyecto

## ¿Cómo mejoró el desarrollo del proyecto al utilizar herramientas de *tracking* y despliegue?

El uso de herramientas de tracking y despliegue mejoró significativamente el desarrollo del proyecto en varios aspectos:

- **Reproducibilidad**: El uso de contenedores Docker permitió crear entornos consistentes y reproducibles, eliminando el clásico problema de "funciona en mi máquina".

- **Separación de responsabilidades**: La arquitectura modular con Airflow para el pipeline ML y FastAPI/Gradio para el despliegue permite que diferentes equipos trabajen independientemente en cada componente sin interferencias.

- **Monitoreo del flujo**: Airflow proporciona una interfaz visual que permite monitorear el estado de cada tarea del pipeline, identificar cuellos de botella y detectar fallos rápidamente, mejorando significativamente el debugging y mantenimiento.

- **Automatización del ciclo de vida**: La integración entre el pipeline de entrenamiento y la aplicación de deployment permite que nuevos modelos estén disponibles automáticamente para predicción sin intervención manual ahorrando mucho tiempo a largo plazo.

## ¿Qué aspectos del despliegue con `Gradio/FastAPI` fueron más desafiantes o interesantes?

Los aspectos más desafiantes e interesantes del despliegue fueron:

1. **Problemas mlflow**: Los problemas más grandes surgieron al intentar usar mlflow. Hubo muchos problemas con la implementación, desde problemas de permisos hasta versiones incompatibles que llevaron a usar muchas soluciones.

2. **Gestión de dependencias entre módulos**: Los modelos y pipelines guardados con pickle contenían referencias a módulos personalizados (`training`, `config`) que no estaban disponibles en el contenedor de FastAPI.

3. **Arquitectura multi-contenedor**: Configurar correctamente la comunicación entre los contenedores de frontend (Gradio) y backend (FastAPI) mediante redes Docker, asegurando que las variables de entorno y endpoints estuvieran correctamente configurados.

## ¿Cómo aporta `Airflow` a la robustez y escalabilidad del pipeline?

1. **Gestión de fallos**: Airflow permite configurar reintentos automáticos, timeouts y manejo de errores a nivel de tarea. Si una tarea falla, no compromete todo el pipeline y puede reintentar o alertar al equipo.

2. **Dependencias explícitas**: El DAG define claramente las dependencias entre tareas, asegurando que cada paso se ejecute en el orden correcto y solo cuando sus dependencias hayan completado exitosamente.

3. **Monitoreo y alertas**: La interfaz web de Airflow proporciona visibilidad en tiempo real del estado de la pipeline.

4. **Paralelización**: Airflow puede ejecutar tareas independientes en paralelo, aprovechando múltiples cores o incluso múltiples máquinas.

5. **Scheduling flexible**: El sistema de scheduling permite ejecutar el pipeline a intervalos regulares (diario, semanal, mensual) o basado en eventos externos (nuevos datos disponibles).

## ¿Qué se podría mejorar en una versión futura del flujo? ¿Qué partes automatizarían más, qué monitorearían o qué métricas agregarían?

### Mejoras de Automatización:

1. **Validación y testing automatizado**:
   - Agregar tests unitarios y de integración que se ejecuten automáticamente en cada cambio del código.
   - Crear un conjunto de datos de validación estático para verificar que nuevos modelos no empeoren en casos críticos.

2. **Gestión automática de modelos**:
   - Implementar versionado semántico automático de modelos con rollback automático si se detectan problemas para que no ocurran 'incendios' por código.

### Monitoreo:

1. **Data drift detection**:
   - Expandir el módulo de drift detection actual para monitorear no solo datos históricos sino también datos en tiempo real (ya que siempre quiero predecir sobre la próxima semana).

2. **Performance del pipeline**:
   - Tiempo de ejecución de cada tarea del DAG de Airflow.
   - Uso de recursos (CPU, memoria, disco) durante el entrenamiento.
   - Tamaño de los datasets procesados en cada etapa.
