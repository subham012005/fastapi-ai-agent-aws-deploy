# Use official AWS Lambda Python base image
FROM public.ecr.aws/lambda/python:3.9

# Copy function code and dependencies
COPY task1.py task2.py task3.py requirements.txt .
COPY document_for_rag ./document_for_rag

# Install dependencies
RUN pip install -r requirements.txt

# Set the CMD to your handler (task3.handler)
CMD [ "task3.handler" ]
