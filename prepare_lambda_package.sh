#!/bin/bash
# Script para preparar un paquete de despliegue para AWS Lambda

# Crear directorio temporal
echo "Creando directorio temporal..."
mkdir -p lambda_package

# Instalar dependencias en el directorio
echo "Instalando dependencias..."
pip install -r requirements.txt --target ./lambda_package

# Copiar archivos del proyecto
echo "Copiando archivos del proyecto..."
cp -r src lambda_package/
cp lambda_function.py lambda_package/

# Crear archivo ZIP
echo "Creando archivo ZIP..."
cd lambda_package
zip -r ../lambda_deployment_package.zip .
cd ..

echo "Paquete creado: lambda_deployment_package.zip"
echo "Tamaño del paquete:"
ls -lh lambda_deployment_package.zip

echo "Ahora debes subir tus archivos de modelo a S3:"
echo "1. Crea un bucket en S3 (si no tienes uno)"
echo "2. Sube tu modelo: aws s3 cp trained_models/price_model.pkl s3://tu-bucket/models/price_model.pkl"
echo "3. Sube tu vectorizador: aws s3 cp trained_models/vectorizer.pkl s3://tu-bucket/models/vectorizer.pkl"
echo "4. Actualiza las variables de entorno MODEL_BUCKET, MODEL_KEY y VECTORIZER_KEY en tu función Lambda"

# Limpiar
echo "¿Quieres eliminar el directorio temporal? (s/n)"
read response
if [ "$response" = "s" ]; then
  echo "Eliminando directorio temporal..."
  rm -rf lambda_package
  echo "Directorio eliminado."
else
  echo "Directorio temporal conservado en: $(pwd)/lambda_package"
fi
