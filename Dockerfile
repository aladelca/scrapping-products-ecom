FROM public.ecr.aws/lambda/python:3.12

# Print Python version for debugging
RUN python --version && pip --version

# Copy requirements and install dependencies
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt || exit 1


# Verify what was installed
RUN pip list

# Copy function code and src directory
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/
COPY src ${LAMBDA_TASK_ROOT}/src/

# Set environment variables for NLTK and spaCy
ENV NLTK_DATA=${LAMBDA_TASK_ROOT}/nltk_data
ENV PYTHONPATH=${LAMBDA_TASK_ROOT}

# Set permissions
RUN chmod 644 ${LAMBDA_TASK_ROOT}/lambda_function.py
RUN find ${LAMBDA_TASK_ROOT}/src -type f -name "*.py" -exec chmod 644 {} \;

# Set the handler (Lambda entry point)
CMD [ "lambda_function.lambda_handler" ]
