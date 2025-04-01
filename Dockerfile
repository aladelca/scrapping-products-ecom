FROM public.ecr.aws/lambda/python:3.9

# Print Python version for debugging
RUN python --version && pip --version

# Copy requirements and install dependencies
COPY requirements-lambda.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-lambda.txt

# Verify what was installed
RUN pip list

# Copy function code and src directory
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/
COPY src ${LAMBDA_TASK_ROOT}/src/

# Set permissions
RUN chmod 644 ${LAMBDA_TASK_ROOT}/lambda_function.py
RUN find ${LAMBDA_TASK_ROOT}/src -type f -name "*.py" -exec chmod 644 {} \;

# Set the handler (Lambda entry point)
CMD [ "lambda_function.lambda_handler" ]
