# ML in Production: Iris Classifier API

This repository contains a complete ML in Production workflow for classifying Iris flowers based on their features. The model is trained using scikit-learn, served using a FastAPI application, and is fully containerized using Docker.

## Project Structure

- `train.py`: Script to train the RandomForest classifier and serialize it.
- `app.py`: FastAPI application exposing the `/predict` endpoint.
- `Dockerfile`: Configuration for containerizing the application.
- `requirements.txt`: Python dependencies.
- `test_api.py`: Local test suite for the FastAPI application.
- `model.pkl`: The serialized, pre-trained ML model (tracked in Git for deployment).

## Local Development & Testing

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Train the Model** (Generates `model.pkl`):
   ```bash
   python train.py
   ```

3. **Run the Application Locally**:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

4. **Test the Application**:
   ```bash
   pytest test_api.py
   ```

## Deployment to Render

To deploy this Docker web service to [Render](https://render.com/), follow these exact steps:

### 1. Push Code to GitHub
Ensure all files, including `model.pkl`, are committed and pushed to a GitHub repository:
```bash
git init
git add .
git commit -m "Initial commit with ML model and FastAPI app"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

### 2. Connect GitHub to Render
1. Log in to your [Render Dashboard](https://dashboard.render.com/).
2. Click on the **New** button and select **Web Service**.
3. Under the "Connect a repository" section, locate and select the GitHub repository you just pushed. If it's your first time, you may need to authenticate and install the Render GitHub app.

### 3. Configure the Web Service
Once the repository is selected, configure the service as follows:
- **Name**: Choose a unique name for your application (e.g., `iris-classifier-api`).
- **Region**: Choose the region closest to your users.
- **Branch**: `main` (or the branch you pushed to).
- **Environment**: Select **Docker** (Render will automatically detect the `Dockerfile`).
- **Instance Type**: Select the **Free** instance (or a paid instance depending on your needs).
- **Advanced Options**: You do not need any special environment variables, but ensure that the internal port Render detects is `8000` (which is configured in our `Dockerfile`).

### 4. Deploy and Monitor
1. Click the **Create Web Service** button at the bottom of the page.
2. Render will begin building the Docker image based on your `Dockerfile` and then deploy it.
3. **Monitor the Live Logs**: You can watch the real-time build and application logs directly in the Render dashboard for your service. The application logs will display the startup process, the successful loading of `model.pkl`, and all incoming requests to the `/predict` endpoint (thanks to our Python `logging` configuration).

Once the deployment is marked as "Live", you will receive a public URL (e.g., `https://iris-classifier-api.onrender.com`).

### 5. Test the Live API
You can test the deployed application by sending a POST request:
```bash
curl -X POST "https://iris-classifier-api.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}'
```
You should receive a response like `{"species":"setosa"}`.
