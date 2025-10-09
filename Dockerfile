# 1. Use an official Python runtime as a parent image
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy ONLY the API requirements file
COPY requirements_api.txt .

# 4. Install the smaller, necessary packages
RUN pip install --no-cache-dir -r requirements_api.txt

# 5. Copy the rest of your application's code into the container
COPY . .

# 6. Expose the port the app runs on
EXPOSE 8000

# 7. Define the command to run your app
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]