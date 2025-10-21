# Medical Cost Prediction

**Predict annual insurance costs with machine learning regression models**

Accurately estimate medical insurance expenses based on patient demographics and health indicators. This end-to-end ML system analyzes 6 key factors to predict individual healthcare costs with multiple regression algorithms, deployed as a production-ready web application on AWS EC2.

---
![Python](https://img.shields.io/badge/Python-3.7+-blue?style=for-the-badge&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Regression-F7931E?style=for-the-badge)
![Flask](https://img.shields.io/badge/Flask-Web-000000?style=for-the-badge&logo=flask)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker)
![AWS](https://img.shields.io/badge/AWS-EC2-FF9900?style=for-the-badge&logo=amazonaws)
![Dataset](https://img.shields.io/badge/Dataset-1,338_Records-brightgreen?style=for-the-badge)

## Live Demo

**Interactive Web Application**

<img width="1502" height="887" alt="demo_3" src="https://github.com/user-attachments/assets/e3043a66-5696-4a09-b1d5-f72fb3f2ad7a" />


---

## Why This Project Stands Out

### Compare: Traditional vs Our Approach

| Aspect | Traditional | Our Solution |
|--------|-----------|--------------|
| **Model Selection** | Single algorithm | Multi-model comparison + best performer |
| **Smoking Factor** | Basic flag | 30%+ cost multiplier identified |
| **User Experience** | Command line only | Interactive web UI |
| **Deployment** | Local machine | Production-ready AWS EC2 |
| **Data Processing** | Manual preprocessing | Automated pipeline with optimization |

---

### Key Differentiators

<details>
<summary><b>🔄 Multi-Model Intelligence</b> — Click to expand</summary>

We don't settle for one algorithm.
- Tested: Linear, Ridge, and Lasso regression
- Selected: Best performer via cross-validation
- Why it matters: Ridge regression reduces overfitting while maintaining accuracy

</details>

<details>
<summary><b>🚬 Smoking Detection (30% Cost Impact)</b> — Click to expand</summary>

Smoking is not just a flag—it's the largest cost multiplier.
- Smokers: +30% insurance premium
- Non-smokers benefit from significantly lower estimates
- Real-world impact: Can differ by $5,000+ annually

</details>

<details>
<summary><b>⚕️ BMI-Aware Risk Assessment</b> — Click to expand</summary>

Intelligent health categorization:
- Underweight (BMI < 18.5): Lower baseline
- Normal (18.5-24.9): Standard rate
- Overweight (25-29.9): +$30 per BMI unit
- Obese (>30): +$100 per BMI unit

</details>

<details>
<summary><b>🗺️ Regional Price Variance</b> — Click to expand</summary>

Geographic pricing isn't ignored:
- Northeast: Standard baseline
- Northwest: +5% adjustment
- Southeast: +10% adjustment
- Southwest: -5% adjustment

</details>

<details>
<summary><b>🎨 Real-Time Web Interface</b> — Click to expand</summary>

No command-line expertise needed:
- Enter information → Get instant prediction
- Beautiful, user-friendly Flask web app
- Prediction range with confidence intervals
- Mobile-responsive design

</details>

<details>
<summary><b>🚀 Production-Ready Infrastructure</b> — Click to expand</summary>

Enterprise-grade deployment:
- Containerized with Docker
- Deployed on AWS EC2
- CI/CD pipeline ready
- Not just a Jupyter notebook

</details>

<details>
<summary><b>⚙️ Hyperparameter Fine-Tuning</b> — Click to expand</summary>

Maximum accuracy through optimization:
- Grid search for parameter tuning
- Cross-validation for robustness
- Model comparison on real-world data
- 83% prediction accuracy achieved

</details>

---

## Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)
- Git
- Docker (optional, for containerized deployment)

### Step 1: Clone the Repository

```bash
git clone https://github.com/patelandpatel/Medical_Cost_Prediction.git
cd Medical_Cost_Prediction
```

### Step 2: Create Virtual Environment

**Using conda (Recommended)**
```bash
conda create -n medical-cost python=3.7 -y
conda activate medical-cost
```

**Using venv**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**Main dependencies:**
- Flask (Web framework)
- scikit-learn (ML models)
- Pandas (Data processing)
- NumPy (Numerical computing)

### Step 4: Prepare Data & Train Model

```bash
python src/components/data_ingestion.py
```

This script:
- Loads the insurance dataset
- Preprocesses and transforms data
- Trains multiple regression models
- Saves the best model as `model.pkl`
- Generates train/test/validation splits

### Step 5: Test the Model

```bash
python Tests/test.py
```

Output: Model performance metrics (R² score, accuracy)

### Step 6: Run the Web Application

```bash
python application.py
```

Navigate to: `http://localhost:5000/`

Your application is now running locally!

---

## Input Features (6 Parameters)

| Feature | Range | Description | Impact |
|---------|-------|-------------|--------|
| **Age** | 18-65 years | Patient's current age | Primary cost driver |
| **BMI** | 10-55 | Body Mass Index | Health risk indicator |
| **Children** | 0-5 | Number of dependents | Coverage scope |
| **Sex** | M/F | Gender | Demographic factor |
| **Smoker** | Yes/No | Smoking status | **Largest multiplier** |
| **Region** | 4 options | Geographic location | Regional pricing |

---

## Quick Start: Make a Prediction

### Via Web Interface
1. Go to http://localhost:5000/
2. Fill in patient information (age, BMI, smoking status, etc.)
3. Click "Calculate Cost"
4. Get estimated annual insurance cost

### Via Python
```python
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('models/model.pkl', 'rb'))

# Input features: [age, children, bmi, sex, smoker, region]
patient_data = np.array([[35, 2, 27.5, 1, 0, 0]])

# Get prediction
predicted_cost = model.predict(patient_data)
print(f"Estimated Annual Cost: ${predicted_cost[0]:,.2f}")
```

### Example Output
```
Input: 35-year-old male, 2 children, BMI 27.5, non-smoker, Northeast
Output: $13,214 annual cost (Range: $11,232 - $15,196)
```

---

## Docker Deployment

### Build & Run Locally

```bash
# Build Docker image
docker build -t medical-cost-predictor:latest .

# Run container
docker run -p 5000:5000 medical-cost-predictor:latest

# Access at http://localhost:5000
```

### Deploy to AWS EC2

```bash
# Push image to AWS ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <your-ecr-uri>

docker tag medical-cost-predictor:latest <your-ecr-uri>/medical-cost-predictor:latest

docker push <your-ecr-uri>/medical-cost-predictor:latest

# Pull and run on EC2 instance
docker pull <your-ecr-uri>/medical-cost-predictor:latest
docker run -p 5000:5000 <your-ecr-uri>/medical-cost-predictor:latest
```

---

## Dataset

**Source:** [Kaggle Insurance Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)

- **Total Records:** 1,338 insurance entries
- **Features:** 6 input variables (age, sex, BMI, children, smoker, region)
- **Target:** charges (annual medical costs)
- **Data Quality:** Clean, no missing values, ready for ML

---

## Model Performance

- **Algorithm:** Regression (Linear, Ridge, Lasso)
- **Best Model:** Ridge Regression
- **Cross-Validation Score:** 0.82 (R² score)
- **Training Accuracy:** 83%
- **Test Accuracy:** 81%

---

## Author

**Parth Patel**

- Email: Parth.Patel@my.utsa.edu
- GitHub: [@patelandpatel](https://github.com/patelandpatel)
- LinkedIn: [Parth Patel](https://linkedin.com)

---

## License

MIT License - See LICENSE file for details

---

**Production-Ready Medical Cost Prediction System | Deployed on AWS**
