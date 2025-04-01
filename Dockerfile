FROM public.ecr.aws/lambda/python:3.9

# Copy requirements and install dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy function code and src directory
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/
COPY src ${LAMBDA_TASK_ROOT}/src/

# Set permissions
RUN chmod 644 ${LAMBDA_TASK_ROOT}/lambda_function.py
RUN find ${LAMBDA_TASK_ROOT}/src -type f -name "*.py" -exec chmod 644 {} \;

# Set the handler (Lambda entry point)
CMD [ "lambda_function.lambda_handler" ]
