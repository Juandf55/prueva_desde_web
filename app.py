import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configure page
st.set_page_config(
    page_title="Spam Email Classification Analysis",
    page_icon="📧",
    layout="wide"
)

# Title and description
st.title("📧 Spam Email Classification Analysis")
st.markdown("### Interactive Machine Learning Application for Email Spam Detection")
st.markdown("---")

# Initialize session state variables
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'tfidf_vectorizer' not in st.session_state:
    st.session_state.tfidf_vectorizer = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a section:",
    ["Data Loading", "Data Exploration", "Preprocessing", "Feature Engineering", 
     "Descriptive Analytics", "Model Training", "Model Evaluation", "Predictions", "Summary"]
)

# Data Loading Section
if page == "Data Loading":
    st.header("📁 Data Loading")
    
    try:
        # Load the dataset
        df = pd.read_csv("sample_data/combined_data.csv", encoding="latin-1", on_bad_lines="skip")
        st.session_state.df = df
        st.session_state.data_loaded = True
        
        st.success("✅ Dataset loaded successfully!")
        
        st.subheader("First 5 rows of the dataset:")
        st.dataframe(df.head())
        
        st.subheader("Dataset Information:")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
    except FileNotFoundError:
        st.error("❌ Dataset file 'combined_data.csv' not found in sample_data folder. Please upload the dataset.")
        st.info("💡 The expected file path is: `sample_data/combined_data.csv`")
        st.session_state.data_loaded = False
    except Exception as e:
        st.error(f"❌ Error loading dataset: {str(e)}")
        st.session_state.data_loaded = False

# Data Exploration Section
elif page == "Data Exploration":
    st.header("🔍 Data Exploration and Preprocessing")
    
    if not st.session_state.data_loaded or st.session_state.df is None:
        st.warning("⚠️ Please load the dataset first from the 'Data Loading' section.")
    else:
        df = st.session_state.df
        
        st.subheader("Dataset Structure Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Shape of the DataFrame:**")
            st.info(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
            
            st.write("**Data Types:**")
            st.dataframe(df.dtypes.to_frame().rename(columns={0: 'Data Type'}))
            
        with col2:
            st.write("**Missing Values:**")
            missing_values = df.isnull().sum()
            st.dataframe(missing_values.to_frame().rename(columns={0: 'Missing Count'}))
            
            st.write("**Label Distribution:**")
            if 'label' in df.columns:
                label_counts = df['label'].value_counts()
                st.dataframe(label_counts.to_frame().rename(columns={'label': 'Count'}))
            else:
                st.error("❌ 'label' column not found in dataset")

# Preprocessing Section
elif page == "Preprocessing":
    st.header("🧹 Text Preprocessing")
    
    if not st.session_state.data_loaded or st.session_state.df is None:
        st.warning("⚠️ Please load the dataset first from the 'Data Loading' section.")
    else:
        df = st.session_state.df.copy()
        
        if 'text' not in df.columns:
            st.error("❌ 'text' column not found in dataset")
        else:
            st.subheader("Before Preprocessing:")
            st.write("Sample text examples:")
            st.dataframe(df[['text']].head(3))
            
            # Text preprocessing
            st.subheader("Applying Text Preprocessing...")
            
            with st.spinner("Processing text data..."):
                # Convert to lowercase
                df['text'] = df['text'].str.lower()
                
                # Remove special characters and punctuation
                df['text'] = df['text'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x)))
                
                # Remove extra whitespace
                df['text'] = df['text'].apply(lambda x: ' '.join(str(x).split()))
            
            st.success("✅ Text preprocessing completed!")
            
            st.subheader("After Preprocessing:")
            st.write("Sample processed text examples:")
            st.dataframe(df[['text']].head(3))
            
            # Update session state
            st.session_state.df = df

# Feature Engineering Section
elif page == "Feature Engineering":
    st.header("⚙️ Feature Engineering")
    
    if not st.session_state.data_loaded or st.session_state.df is None:
        st.warning("⚠️ Please load and preprocess the dataset first.")
    else:
        df = st.session_state.df
        
        st.subheader("TF-IDF Vectorization")
        st.write("Transforming text data into numerical features using Term Frequency-Inverse Document Frequency (TF-IDF)")
        
        with st.spinner("Applying TF-IDF vectorization..."):
            # Initialize TfidfVectorizer
            tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            
            # Fit and transform the text data
            tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])
            
            # Store in session state
            st.session_state.tfidf_vectorizer = tfidf_vectorizer
            st.session_state.tfidf_matrix = tfidf_matrix
        
        st.success("✅ TF-IDF vectorization completed!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("TF-IDF Matrix Shape - Rows", tfidf_matrix.shape[0])
        with col2:
            st.metric("TF-IDF Matrix Shape - Features", tfidf_matrix.shape[1])
        with col3:
            sparsity = (1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])) * 100
            st.metric("Sparsity", f"{sparsity:.2f}%")
        
        st.subheader("What is TF-IDF?")
        st.info("""
        **TF-IDF (Term Frequency-Inverse Document Frequency)** is a numerical statistic that reflects how important a word is to a document in a collection of documents. It increases proportionally to the number of times a word appears in a document but is offset by the frequency of the word in the corpus, which helps to adjust for the fact that some words appear more frequently in general.
        
        - **TF (Term Frequency)**: How frequently a term appears in a document
        - **IDF (Inverse Document Frequency)**: How rare or common a term is across all documents
        - **TF-IDF**: TF × IDF - gives higher weights to terms that are frequent in a document but rare across the corpus
        """)

# Descriptive Analytics Section
elif page == "Descriptive Analytics":
    st.header("📊 Descriptive Analytics")
    
    if not st.session_state.data_loaded or st.session_state.df is None:
        st.warning("⚠️ Please load the dataset first.")
    else:
        df = st.session_state.df
        
        # Calculate text length
        df['text_length'] = df['text'].apply(len)
        
        st.subheader("Email Length Analysis")
        
        # Calculate average lengths
        if 'label' in df.columns:
            avg_length_spam = df[df['label'] == 1]['text_length'].mean()
            avg_length_ham = df[df['label'] == 0]['text_length'].mean()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Average Length - Spam Emails", f"{avg_length_spam:.2f} characters")
            with col2:
                st.metric("Average Length - Ham Emails", f"{avg_length_ham:.2f} characters")
            
            # Visualization
            st.subheader("Text Length Distribution")
            
            fig = px.histogram(
                df, 
                x='text_length', 
                color='label',
                nbins=50,
                title="Distribution of Email Text Length by Category",
                labels={'text_length': 'Text Length (characters)', 'label': 'Email Type'},
                color_discrete_map={0: 'lightblue', 1: 'salmon'}
            )
            fig.update_layout(
                xaxis_title="Text Length (characters)",
                yaxis_title="Frequency",
                legend_title="Email Type"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Box plot for better comparison
            st.subheader("Text Length Comparison (Box Plot)")
            fig_box = px.box(
                df, 
                x='label', 
                y='text_length',
                title="Email Text Length Distribution by Category",
                labels={'label': 'Email Type (0=Ham, 1=Spam)', 'text_length': 'Text Length (characters)'},
                color='label',
                color_discrete_map={0: 'lightblue', 1: 'salmon'}
            )
            st.plotly_chart(fig_box, use_container_width=True)
            
            # Additional statistics
            st.subheader("Detailed Statistics")
            stats_df = df.groupby('label')['text_length'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
            stats_df.index = ['Ham (0)', 'Spam (1)']
            st.dataframe(stats_df)
            
        else:
            st.error("❌ 'label' column not found for analysis")

# Model Training Section
elif page == "Model Training":
    st.header("🤖 Model Selection and Training")
    
    if not st.session_state.data_loaded or st.session_state.df is None:
        st.warning("⚠️ Please complete feature engineering first.")
    elif not hasattr(st.session_state, 'tfidf_matrix'):
        st.warning("⚠️ Please complete feature engineering first.")
    else:
        df = st.session_state.df
        tfidf_matrix = st.session_state.tfidf_matrix
        
        st.subheader("Data Splitting")
        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.25, 0.05)
        random_state = st.number_input("Random State", value=42, min_value=1)
        
        if st.button("Train Model"):
            with st.spinner("Training Logistic Regression model..."):
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(
                    tfidf_matrix, df['label'], 
                    test_size=test_size, 
                    random_state=random_state
                )
                
                # Train the model
                model = LogisticRegression(max_iter=1000)
                model.fit(X_train, y_train)
                
                # Store in session state
                st.session_state.model = model
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.model_trained = True
                
            st.success("✅ Model trained successfully!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Training Set Size", len(X_train.toarray()) if hasattr(X_train, 'toarray') else len(X_train))
            with col2:
                st.metric("Test Set Size", len(X_test.toarray()) if hasattr(X_test, 'toarray') else len(X_test))
            with col3:
                st.metric("Features", X_train.shape[1] if hasattr(X_train, 'shape') else 0)
            with col4:
                st.metric("Model Type", "Logistic Regression")

# Model Evaluation Section
elif page == "Model Evaluation":
    st.header("📈 Model Evaluation")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first from the 'Model Training' section.")
    else:
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        
        # Make predictions
        if model is not None:
            y_pred = model.predict(X_test)
        else:
            st.error("Model not available")
            st.stop()
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        st.subheader("Model Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{accuracy:.4f}")
        with col2:
            st.metric("Precision", f"{precision:.4f}")
        with col3:
            st.metric("Recall", f"{recall:.4f}")
        with col4:
            st.metric("F1-Score", f"{f1:.4f}")
        
        st.subheader("Confusion Matrix")
        
        # Create confusion matrix visualization
        fig = px.imshow(
            conf_matrix,
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=['Ham', 'Spam'],
            y=['Ham', 'Spam'],
            color_continuous_scale='Blues',
            text_auto=True,
            title="Confusion Matrix"
        )
        fig.update_layout(width=500, height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Confusion Matrix Interpretation")
        st.info(f"""
        **Confusion Matrix Results:**
        - True Negatives (Ham correctly classified): {conf_matrix[0,0]}
        - False Positives (Ham classified as Spam): {conf_matrix[0,1]}
        - False Negatives (Spam classified as Ham): {conf_matrix[1,0]}
        - True Positives (Spam correctly classified): {conf_matrix[1,1]}
        
        **Purpose of Confusion Matrix:**
        The confusion matrix provides a detailed breakdown of correct and incorrect classifications, helping us understand where the model makes mistakes and evaluate its performance beyond simple accuracy.
        """)

# Predictions Section
elif page == "Predictions":
    st.header("🔮 Prediction and Interpretation")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train the model first from the 'Model Training' section.")
    else:
        model = st.session_state.model
        tfidf_vectorizer = st.session_state.tfidf_vectorizer
        
        st.subheader("Test New Email Examples")
        
        # Predefined examples
        st.write("**Predefined Test Examples:**")
        
        predefined_examples = [
            "Congratulations! You've won a free vacation! Click here to claim your prize.",
            "Meeting reminder: Our team meeting is scheduled for tomorrow at 10 AM in the conference room.",
            "Urgent: Your account has been compromised. Please verify your details immediately.",
            "Just checking in to see how you're doing. Hope you have a great day!",
            "Limited time offer: Get 50% off on all products this week only!",
            "Regarding your recent inquiry, please find the requested document attached."
        ]
        
        if st.button("Test Predefined Examples"):
            # Preprocess the examples
            processed_examples = []
            for email in predefined_examples:
                processed_email = email.lower()
                processed_email = re.sub(r'[^a-zA-Z\s]', '', processed_email)
                processed_email = ' '.join(processed_email.split())
                processed_examples.append(processed_email)
            
            # Transform and predict
            if tfidf_vectorizer is not None and model is not None:
                examples_tfidf = tfidf_vectorizer.transform(processed_examples)
                predictions = model.predict(examples_tfidf)
                probabilities = model.predict_proba(examples_tfidf)
            else:
                st.error("Model or vectorizer not available")
                st.stop()
            
            st.subheader("Prediction Results")
            
            results_df = pd.DataFrame({
                'Email Text': predefined_examples,
                'Predicted Label': ['Spam' if pred == 1 else 'Ham' for pred in predictions],
                'Spam Probability': [f"{prob[1]:.3f}" for prob in probabilities],
                'Ham Probability': [f"{prob[0]:.3f}" for prob in probabilities]
            })
            
            st.dataframe(results_df, use_container_width=True)
        
        st.markdown("---")
        
        # Custom email testing
        st.subheader("Test Your Own Email")
        
        custom_email = st.text_area(
            "Enter an email text to classify:",
            placeholder="Type or paste an email here...",
            height=100
        )
        
        if st.button("Classify Email") and custom_email:
            # Preprocess the custom email
            processed_email = custom_email.lower()
            processed_email = re.sub(r'[^a-zA-Z\s]', '', processed_email)
            processed_email = ' '.join(processed_email.split())
            
            # Transform and predict
            if tfidf_vectorizer is not None and model is not None:
                email_tfidf = tfidf_vectorizer.transform([processed_email])
                prediction = model.predict(email_tfidf)[0]
                probability = model.predict_proba(email_tfidf)[0]
            else:
                st.error("Model or vectorizer not available")
                st.stop()
            
            # Display result
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.error(f"🚨 **SPAM** (Confidence: {probability[1]:.1%})")
                else:
                    st.success(f"✅ **HAM** (Confidence: {probability[0]:.1%})")
            
            with col2:
                st.write("**Probability Breakdown:**")
                st.write(f"Ham: {probability[0]:.3f}")
                st.write(f"Spam: {probability[1]:.3f}")

# Summary Section
elif page == "Summary":
    st.header("📋 Analysis Summary")
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please complete the analysis first.")
    else:
        df = st.session_state.df
        
        st.subheader("Key Findings - Lab Questions & Answers")
        
        # Question 1: Dataset size and missing values
        st.write("**1. How many emails did the dataset contain? Are there any missing values?**")
        if df is not None:
            missing_values = df.isnull().sum()
            total_missing = missing_values.sum()
            st.info(f"The dataset contains **{df.shape[0]} emails**. Missing values: **{total_missing}** (Details: {dict(missing_values)})")
        
        # Question 2: Class imbalance
        st.write("**2. Were there any imbalances in the emails? (spam vs ham distribution)**")
        if df is not None and 'label' in df.columns:
            label_counts = df['label'].value_counts()
            ham_count = label_counts.get(0, 0)
            spam_count = label_counts.get(1, 0)
            ham_pct = (ham_count / len(df)) * 100
            spam_pct = (spam_count / len(df)) * 100
            st.info(f"Ham emails: **{ham_count}** ({ham_pct:.1f}%), Spam emails: **{spam_count}** ({spam_pct:.1f}%). The dataset shows {'significant imbalance' if abs(ham_pct - spam_pct) > 20 else 'reasonable balance'}.")
        
        # Question 3: Text length comparison
        st.write("**3. What are the average characters for ham vs spam emails? Which was longer?**")
        if df is not None and 'text_length' in df.columns:
            avg_ham = df[df['label'] == 0]['text_length'].mean()
            avg_spam = df[df['label'] == 1]['text_length'].mean()
            longer_type = "Spam" if avg_spam > avg_ham else "Ham"
            st.info(f"Average characters - Ham: **{avg_ham:.0f}**, Spam: **{avg_spam:.0f}**. **{longer_type}** emails are longer on average.")
        
        # Question 4: TF-IDF explanation
        st.write("**4. What is TF-IDF? Provide a few sentences explaining this.**")
        st.info("""
        **TF-IDF (Term Frequency-Inverse Document Frequency)** is a numerical statistic that reflects the importance of a word in a document relative to a collection of documents. TF measures how frequently a term appears in a document, while IDF measures how rare or common a term is across all documents. The TF-IDF score increases proportionally to the number of times a word appears in a document but is offset by its frequency in the corpus, helping to identify words that are particularly characteristic of specific documents while downweighting common words that appear everywhere.
        """)
        
        # Question 5: Model performance
        st.write("**5. Using the trained Logistic Regression model, how accurate was the evaluation?**")
        if st.session_state.model_trained and st.session_state.model is not None:
            model = st.session_state.model
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            y_pred = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            st.info(f"Model Performance - Accuracy: **{accuracy:.4f}** ({accuracy*100:.2f}%), Precision: **{precision:.4f}**, Recall: **{recall:.4f}**, F1-Score: **{f1:.4f}**")
        else:
            st.warning("Model not trained yet.")
        
        # Question 6: Confusion Matrix purpose
        st.write("**6. What is the purpose of the Confusion Matrix?**")
        st.info("""
        The **Confusion Matrix** provides a detailed breakdown of correct and incorrect classifications made by the model. It shows True Positives (spam correctly identified), True Negatives (ham correctly identified), False Positives (ham incorrectly classified as spam), and False Negatives (spam incorrectly classified as ham). This matrix helps evaluate model performance beyond simple accuracy, revealing specific types of errors and helping identify whether the model has bias toward classifying emails as spam or ham.
        """)
        
        # Question 7: Prediction accuracy
        st.write("**7. Did the new email examples predict correctly/accurately?**")
        st.info("""
        The predefined email examples were designed to test the model's ability to identify obvious spam patterns (prizes, urgent account warnings, limited offers) versus legitimate communications (meeting reminders, personal check-ins, business inquiries). Review the predictions in the 'Predictions' section to evaluate accuracy against expected classifications.
        """)
        
        # Question 8: Real-world applications
        st.write("**8. What would be the next steps with this model? How could cybersecurity professionals use this?**")
        st.info("""
        **Next Steps:** 1) Improve the model with more sophisticated algorithms (ensemble methods, neural networks), 2) Expand feature engineering (metadata analysis, sender reputation), 3) Implement continuous learning to adapt to new spam patterns, 4) Deploy in production with real-time classification.
        
        **Cybersecurity Applications:** Email security gateways can use this model to automatically filter spam before it reaches users' inboxes, reducing phishing attempts and malicious content. The model can be integrated into email servers to provide real-time classification, helping organizations protect against social engineering attacks and maintaining productivity by reducing unwanted emails.
        
        **Simple Explanation:** Think of this model as a smart email assistant that has learned to recognize patterns in thousands of emails. Just like a human can spot suspicious emails by looking for certain keywords or phrases, this computer model does the same thing but much faster and more consistently, helping keep your inbox clean and secure.
        """)

# Footer
st.markdown("---")
st.markdown("### 📧 Spam Email Classification Analysis - Interactive ML Application")
st.markdown("Navigate through the different sections using the sidebar to explore the complete machine learning workflow.")
