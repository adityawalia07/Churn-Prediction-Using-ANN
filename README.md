# Customer Churn Prediction Dashboard

![Churn Prediction Dashboard](https://github.com/username/customer-churn-prediction/raw/main/screenshot.png)

> An interactive web application that predicts customer churn probability using machine learning.

## 📋 Overview

This Streamlit application provides a user-friendly interface for predicting the likelihood of customer churn based on various customer attributes. The underlying prediction model is a neural network built with TensorFlow that analyzes customer data to identify patterns associated with customer attrition.

## ✨ Features

- **Interactive Dashboard**: Input customer details through an intuitive interface
- **Real-time Predictions**: Instantly calculate churn probability with a single click
- **Visual Results**: View prediction results through color-coded indicators and gauge charts
- **Actionable Insights**: Receive tailored recommendations based on churn risk level
- **User-friendly Interface**: Organized layout with logical grouping of input parameters

## 🛠️ Technologies Used

- **Streamlit**: For the web application interface
- **TensorFlow**: For the underlying neural network model
- **scikit-learn**: For data preprocessing
- **Pandas**: For data manipulation
- **Matplotlib**: For data visualization
- **Pickle**: For model serialization

## 📦 Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/username/customer-churn-prediction.git
   cd customer-churn-prediction
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## 📄 Required Files

Make sure you have the following files in your project directory:

- `model.h5`: The trained TensorFlow model
- `label_encoder_gender.pkl`: Pickled label encoder for gender
- `onehot_encoder_geo.pkl`: Pickled one-hot encoder for geography
- `scaler.pkl`: Pickled standard scaler for numerical features

## 🚀 Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Access the application in your web browser at `http://localhost:8501`

3. Enter customer details in the provided input fields:
   - Personal information (Geography, Gender, Age)
   - Account information (Tenure, Number of Products)
   - Financial details (Balance, Salary, Credit Score)
   - Additional factors (Credit Card, Activity Status)

4. Click the "Predict Churn" button to view the prediction results

## 🧠 Model Information

The application uses a neural network model trained on historical customer data to predict churn probability. Key predictors include:

- Account balance
- Customer age
- Geographic location
- Activity level
- Credit score
- Product utilization

## 🔧 Customization

You can customize the application by:

1. **Using your own model**: Replace `model.h5` with your trained model
2. **Modifying input parameters**: Edit the input fields in the Streamlit app
3. **Changing visual elements**: Update the styling in the custom CSS section
4. **Adding new features**: Extend the codebase with additional visualization or analysis tools

## 📊 Sample Output

The application provides:

- Churn probability percentage
- Risk classification (High/Low)
- Visual gauge showing probability
- Recommendations based on risk level

## 📝 Training Your Own Model

If you want to train your own churn prediction model, follow these steps:

1. Prepare your customer data with relevant features
2. Preprocess the data (encoding categorical variables, scaling numerical features)
3. Train a neural network or other ML model
4. Save the model and preprocessing objects as described in the Required Files section

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

Your Name - [@twitter_handle](https://twitter.com/twitter_handle) - email@example.com

Project Link: [https://github.com/username/customer-churn-prediction](https://github.com/username/customer-churn-prediction)

---

Made with ❤️ by [Your Name]
