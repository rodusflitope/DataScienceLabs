# Respuestas Docker - Lab 8

## Análisis de Recursos del Contenedor


Podemos ver que los contenedores son muy livianos pues gastan 0.2% de CPU, 2% de memoria y sólo 5kB de escritura y lectura.

---

## Preguntas de Smapina

### ¿Cómo se diferencia Docker de una máquina virtual (VM)?

Las VM requieren de copia completa del sistema operativo y los contenedores usan el mismo que usa el host. Las VM necesitan hipervisores mientras que los contenedores son gestionados por el motor de contenedores y es mucho más ligero. Por esto también la velocidad para iniciar un contenedor es mucho más rápida que para una VM.

### ¿Cuál es la diferencia entre usar Docker y ejecutar la aplicación directamente en el sistema local?

Docker encapsula la aplicación con todas sus dependencias en un contenedor aislado, mientras que ejecutar localmente depende del entorno del sistema que lo ejecuta. Gracias a esto Docker funciona igual (fuera de hardware) para todas las personas que ejecuten el programa en su máquina.

### ¿Cómo asegura Docker la consistencia entre diferentes entornos de desarrollo y producción?

Docker usa el mismo Dockerfile e imagen para todos los entornos. El contenedor incluye las dependencias exactas especificadas en requirements.txt, eliminando problemas de "funciona en mi máquina".

### ¿Cómo se gestionan los volúmenes en Docker para la persistencia de datos?

Los volúmenes son directorios del host que se montan en el contenedor. Permiten que los datos persistan aunque el contenedor se elimine.

### ¿Qué son Dockerfile y docker-compose.yml, y cuál es su propósito?

Dockerfile es un archivo de texto que define cómo construir una imagen Docker paso a paso (imagen base, dependencias, archivos a copiar, comandos). docker-compose.yml orquesta múltiples contenedores, define servicios, redes y volúmenes, simplificando el despliegue de aplicaciones con varios componentes.