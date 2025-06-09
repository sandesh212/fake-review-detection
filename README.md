# Fake Review Detection and Feedback System

This project implements a **Fake Review Detection and Feedback System** using Natural Language Processing (NLP) techniques. It leverages a Long Short-Term Memory (LSTM) model to classify product reviews as fake (computer-generated, labeled `CG`) or genuine (human-written, labeled `OR`). For genuine reviews, it uses OpenAI’s GPT-3.5 model to generate product improvement suggestions, aiding e-commerce platforms in maintaining review authenticity and gathering actionable feedback.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Model Architecture](#model-architecture)
- [Feedback System](#feedback-system)
- [Example](#example)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview
Fake reviews erode consumer trust in online marketplaces. This project addresses this issue by detecting fake reviews using an LSTM-based deep learning model trained on a review dataset. For genuine reviews, it integrates OpenAI’s GPT-3.5 to provide automated suggestions for product improvements, enabling businesses to enhance their offerings based on authentic customer feedback.

## Features
- **Review Classification**: Identifies fake vs. genuine reviews using an LSTM model.
- **Feedback Generation**: Provides product improvement suggestions for genuine reviews via GPT-3.5.
- **Data Visualization**: Displays the distribution of review categories using a bar plot.
- **Data Preprocessing**: Tokenizes and pads review text for model compatibility.
- **Model Persistence**: Saves the trained LSTM model for reuse.

## Dataset
The project uses the `fake_reviews_dataset.csv` dataset, which includes:
- `category`: Product category (e.g., `Home_and_Kitchen_5`).
- `rating`: Numerical rating (1.0 to 5.0).
- `label`: Binary label (`CG` for fake, `OR` for genuine).
- `review`: Review text (originally `text_`, renamed in code).

**Note**: The dataset is not included in this repository due to size constraints. Users must obtain `fake_reviews_dataset.csv` separately and place it in the project directory. Alternatively, adapt the code to use a similar dataset with the same structure.

## Requirements
- **Python**: Version 3.12
- **Libraries**:
  - `pandas` (data manipulation)
  - `matplotlib` (visualization)
  - `numpy` (numerical operations)
  - `scikit-learn` (train-test split)
  - `tensorflow>=2.16.1` (LSTM model)
  - `openai` (GPT-3.5 integration)
- **OpenAI API Key**: Required for GPT-3.5 feedback generation.
- **Hardware**: A machine with sufficient memory to train the LSTM model (GPU optional for faster training).

## Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/sandesh212/fake-review-detection.git
   cd fake-review-detection
   ```

2. **Create a Virtual Environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Ensure `requirements.txt` is in the project directory, then run:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up OpenAI API Key**:
   - Obtain an API key from [OpenAI](https://platform.openai.com/account/api-keys).
   - Set it as an environment variable:
     ```bash
     export OPENAI_API_KEY='your-api-key'  # On Windows: set OPENAI_API_KEY=your-api-key
     ```

5. **Place the Dataset**:
   - Obtain `fake_reviews_dataset.csv` and place it in the project directory.
   - Ensure the notebook uses a relative path: `pd.read_csv("fake_reviews_dataset.csv")`.

## Usage
1. **Launch Jupyter Notebook**:
   ```bash
   jupyter notebook FakeReviewDetection.ipynb
   ```

2. **Execute the Notebook**:
   - Run the cells in sequence to:
     - Load and preprocess the dataset.
     - Visualize review category distribution.
     - Train and evaluate the LSTM model.
     - Test the review classification and feedback system.

3. **Classify a Review and Get Feedback**:
   Use the `handle_customer_review` function to classify a review and retrieve feedback:
   ```python
   sample_review = "This is one of the coolest screensavers I have ever seen, the fish move realistically, the environments look real, and the graphics are stunning."
   print(handle_customer_review(sample_review))
   ```

## Project Structure
```
fake-review-detection