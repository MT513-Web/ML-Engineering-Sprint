# Use official lightweight Python image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files and saved models into the container
COPY . .

# Expose the port for FastAPI application
EXPOSE 8000

# Command to run the FastAPI server inside the container
CMD ["uvicorn", "day_20_fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]