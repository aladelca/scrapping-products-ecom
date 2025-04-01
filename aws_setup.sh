#!/bin/bash
# Script para configurar la infraestructura AWS necesaria para el despliegue continuo

set -e

# Variables (modificar según tus necesidades)
AWS_REGION="us-east-1"
ECR_REPOSITORY_NAME="price-predictor"
LAMBDA_FUNCTION_NAME="predict_description"
S3_BUCKET_NAME="your-model-bucket-name"  # Para almacenar los modelos
IAM_ROLE_NAME="lambda-price-predictor-role"
MODEL_S3_KEY="models/price_model.pkl"
VECTORIZER_S3_KEY="models/vectorizer.pkl"

# Comprobar si AWS CLI está instalado
if ! command -v aws &> /dev/null; then
    echo "Error: AWS CLI no está instalado. Por favor, instala AWS CLI primero."
    exit 1
fi

# 1. Crear el repositorio ECR si no existe
echo "Creando repositorio ECR: $ECR_REPOSITORY_NAME"
aws ecr describe-repositories --repository-names $ECR_REPOSITORY_NAME --region $AWS_REGION 2>/dev/null || \
aws ecr create-repository --repository-name $ECR_REPOSITORY_NAME --region $AWS_REGION

# 2. Crear el bucket S3 para los modelos si no existe
echo "Creando bucket S3: $S3_BUCKET_NAME"
aws s3api head-bucket --bucket $S3_BUCKET_NAME 2>/dev/null || \
aws s3 mb s3://$S3_BUCKET_NAME --region $AWS_REGION

# 3. Subir modelos al bucket S3
echo "Subiendo modelos al bucket S3"
MODEL_PATH="trained_models/price_model.pkl"
VECTORIZER_PATH="trained_models/vectorizer.pkl"

if [ -f "$MODEL_PATH" ]; then
    aws s3 cp $MODEL_PATH s3://$S3_BUCKET_NAME/$MODEL_S3_KEY
    echo "Modelo subido: $MODEL_PATH -> s3://$S3_BUCKET_NAME/$MODEL_S3_KEY"
else
    echo "Advertencia: No se encuentra el archivo del modelo en $MODEL_PATH"
fi

if [ -f "$VECTORIZER_PATH" ]; then
    aws s3 cp $VECTORIZER_PATH s3://$S3_BUCKET_NAME/$VECTORIZER_S3_KEY
    echo "Vectorizador subido: $VECTORIZER_PATH -> s3://$S3_BUCKET_NAME/$VECTORIZER_S3_KEY"
else
    echo "Advertencia: No se encuentra el archivo del vectorizador en $VECTORIZER_PATH"
fi

# 4. Crear rol IAM para Lambda
echo "Creando rol IAM para Lambda: $IAM_ROLE_NAME"
IAM_ROLE_ARN=$(aws iam get-role --role-name $IAM_ROLE_NAME --query 'Role.Arn' --output text 2>/dev/null || echo "")

if [ -z "$IAM_ROLE_ARN" ]; then
    # Crear documento de política de confianza
    cat > trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": {
        "Service": "lambda.amazonaws.com"
      },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

    # Crear el rol
    IAM_ROLE_ARN=$(aws iam create-role \
        --role-name $IAM_ROLE_NAME \
        --assume-role-policy-document file://trust-policy.json \
        --query 'Role.Arn' \
        --output text)

    # Añadir políticas necesarias
    aws iam attach-role-policy \
        --role-name $IAM_ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

    aws iam attach-role-policy \
        --role-name $IAM_ROLE_NAME \
        --policy-arn arn:aws:iam::aws:policy/AmazonS3ReadOnlyAccess

    # Esperar a que el rol esté disponible
    echo "Esperando a que el rol IAM esté disponible..."
    sleep 20

    # Limpiar
    rm trust-policy.json
else
    echo "Rol IAM ya existe: $IAM_ROLE_ARN"
fi

# 5. Construir la imagen Docker inicial
echo "Construyendo imagen Docker inicial"
ECR_URI=$(aws ecr describe-repositories \
    --repository-names $ECR_REPOSITORY_NAME \
    --query 'repositories[0].repositoryUri' \
    --output text)

aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_URI

# Construir y subir la imagen
INITIAL_TAG="initial"
docker build -t $ECR_URI:$INITIAL_TAG .
docker push $ECR_URI:$INITIAL_TAG

# 6. Crear la función Lambda
echo "Creando función Lambda: $LAMBDA_FUNCTION_NAME"
LAMBDA_EXISTS=$(aws lambda list-functions \
    --query "Functions[?FunctionName=='$LAMBDA_FUNCTION_NAME'].FunctionName" \
    --output text)

if [ -z "$LAMBDA_EXISTS" ]; then
    aws lambda create-function \
        --function-name $LAMBDA_FUNCTION_NAME \
        --package-type Image \
        --code ImageUri=$ECR_URI:$INITIAL_TAG \
        --role $IAM_ROLE_ARN \
        --timeout 30 \
        --memory-size 512 \
        --environment "Variables={MODEL_BUCKET=$S3_BUCKET_NAME,MODEL_KEY=$MODEL_S3_KEY,VECTORIZER_KEY=$VECTORIZER_S3_KEY}"

    echo "Función Lambda creada con éxito"
else
    echo "La función Lambda ya existe, actualizando configuración"

    # Actualizar la configuración de la función Lambda
    aws lambda update-function-configuration \
        --function-name $LAMBDA_FUNCTION_NAME \
        --timeout 30 \
        --memory-size 512 \
        --environment "Variables={MODEL_BUCKET=$S3_BUCKET_NAME,MODEL_KEY=$MODEL_S3_KEY,VECTORIZER_KEY=$VECTORIZER_S3_KEY}"

    # Actualizar el código de la función Lambda
    aws lambda update-function-code \
        --function-name $LAMBDA_FUNCTION_NAME \
        --image-uri $ECR_URI:$INITIAL_TAG

    echo "Función Lambda actualizada con éxito"
fi

# 7. Crear un API Gateway (opcional, si quieres exponer el endpoint)
echo "¿Deseas crear un API Gateway para exponer tu función Lambda? (s/n)"
read CREATE_API

if [ "$CREATE_API" = "s" ]; then
    # Crear un REST API
    API_ID=$(aws apigateway create-rest-api \
        --name "Price Predictor API" \
        --query 'id' \
        --output text)

    # Obtener el ID del recurso raíz
    ROOT_RESOURCE_ID=$(aws apigateway get-resources \
        --rest-api-id $API_ID \
        --query 'items[0].id' \
        --output text)

    # Crear un recurso /predict
    RESOURCE_ID=$(aws apigateway create-resource \
        --rest-api-id $API_ID \
        --parent-id $ROOT_RESOURCE_ID \
        --path-part "predict" \
        --query 'id' \
        --output text)

    # Crear método POST
    aws apigateway put-method \
        --rest-api-id $API_ID \
        --resource-id $RESOURCE_ID \
        --http-method POST \
        --authorization-type NONE

    # Integrar con Lambda
    aws apigateway put-integration \
        --rest-api-id $API_ID \
        --resource-id $RESOURCE_ID \
        --http-method POST \
        --type AWS_PROXY \
        --integration-http-method POST \
        --uri arn:aws:apigateway:$AWS_REGION:lambda:path/2015-03-31/functions/arn:aws:lambda:$AWS_REGION:$(aws sts get-caller-identity --query 'Account' --output text):function:$LAMBDA_FUNCTION_NAME/invocations

    # Dar permiso a API Gateway para invocar Lambda
    aws lambda add-permission \
        --function-name $LAMBDA_FUNCTION_NAME \
        --statement-id apigateway-test \
        --action lambda:InvokeFunction \
        --principal apigateway.amazonaws.com \
        --source-arn "arn:aws:execute-api:$AWS_REGION:$(aws sts get-caller-identity --query 'Account' --output text):$API_ID/*/POST/predict"

    # Desplegar la API
    aws apigateway create-deployment \
        --rest-api-id $API_ID \
        --stage-name prod

    # Obtener la URL de la API
    API_URL="https://$API_ID.execute-api.$AWS_REGION.amazonaws.com/prod/predict"

    echo "API Gateway creado con éxito"
    echo "Endpoint de la API: $API_URL"
    echo "Ejemplo de uso:"
    echo "curl -X POST $API_URL -H 'Content-Type: application/json' -d '{\"product_description\":\"Samsung Galaxy A54 128GB 8GB RAM 5G Black Smartphone\"}'"
fi

echo "Configuración completada con éxito!"
echo "Ahora puedes configurar los secretos de GitHub Actions:"
echo "1. Ve a tu repositorio en GitHub"
echo "2. Navega a Settings > Secrets > New repository secret"
echo "3. Añade los siguientes secretos:"
echo "   - AWS_ACCESS_KEY_ID: Tu clave de acceso de AWS"
echo "   - AWS_SECRET_ACCESS_KEY: Tu clave secreta de AWS"
echo ""
echo "Después, cuando hagas push a la rama 'main', GitHub Actions automáticamente desplegará tu código en AWS Lambda."
