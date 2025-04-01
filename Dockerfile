FROM public.ecr.aws/lambda/python:3.9

# Print Python version for debugging
RUN python --version && pip --version

# Copy requirements and install dependencies
COPY requirements-lambda.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements-lambda.txt

# Pre-download NLTK data
RUN mkdir -p /tmp/nltk_data
RUN python -m nltk.downloader -d /tmp/nltk_data punkt stopwords wordnet
# Copy the NLTK data to a location that will be included in the image
RUN cp -r /tmp/nltk_data ${LAMBDA_TASK_ROOT}/nltk_data

# Verify spaCy model installation with error handling
RUN python -c "import sys; import spacy; print('spaCy version:', spacy.__version__); try: nlp = spacy.load('es_core_news_sm'); print('Model loaded successfully:', nlp.meta['name']); except Exception as e: print('Error loading model:', e, file=sys.stderr); sys.exit(1)"

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
