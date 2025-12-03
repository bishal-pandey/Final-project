# Sentiment Analysis on Product Reviews
## _Deep Learning Model using custom LSTM + Attention Mechanism_

This project builds an end-to-end sentiment analysis system that classifies  product reviews into Positive, Negative, or Neutral sentiments. It uses a custom LSTM architecture enhanced with an Attention Layer, trained on 19k+ reviews.

### _Features_
- Web-scraped dataset: 4k Flipkart reviews + Kaggle dataset.  
- Custom **LSTM + Attention** architecture for improved performance.  
- Multi-class sentiment classification (Positive, Negative, Neutral).  
- Complete NLP preprocessing pipeline: text cleaning, tokenization, padding, and embeddings.  
- Evaluation metrics: **Accuracy, Precision, Recall, F1-score**.  
- Web interface for real-time prediction.  


## _Tools Used_
- Programming Language: Python 3.x.
- Deep Learning Framework: PyTorch (for model and training) 
- Data Handling: Pandas for CSV handling, NumPy for numerical operations.
- Preprocessing: python libraries re, String, emoji.
- Web Scraping: Python Selenium for collecting Flipkart reviews.
- Visualization: Matplotlib for plotting confusion matrices and metric curves.
- Evaluation: sklearn for accuracy, precision, recall, F1 calculations.

  ## _Evaluation_

| Embedding | Hidden Size | Learning Rate | Attention           | Accuracy | Precision | Recall | F1 Score |
|-----------|------------|---------------|-------------------|---------|-----------|--------|----------|
| 100       | 64         | 0.0004        | Attention         | **0.98**| **0.98**  | **0.98**| **0.98** ✅| 
| 100       | 64         | 0.001         | Attention          | 0.95    | 0.95      | 0.95   | 0.95     |                
| 50        | 64         | 0.0004        | Attention          | 0.91    | 0.91      | 0.91   | 0.91     |                  
| 100       | 64         | 0.0004        | Attention Disabled | 0.90    | 0.90      | 0.90   | 0.90     |                  


For Logistic Regression and SVM, the text data were transformed into numeric features
by TF-IDF vectorization

| Model                        | Accuracy | Precision | Recall | F1_score |
|-------------------------------|---------|-----------|--------|----------|
| Logistic Regression           | 0.70    | 0.69      | 0.70   | 0.69     |
| SVM                           | 0.71    | 0.70      | 0.71   | 0.70     |
| Custom LSTM + Attention       | **0.98**| **0.98**  | **0.98**| **0.98** ✅|

**Implemented model:** Custom LSTM + Attention (100 embedding, 64 hidden size, learning rate 0.0004, attention enabled)

## Project Pipeline

1. **Preprocessing:** Lowercasing, punctuation & special character removal, contraction expansion, stopword removal, tokenization.  
2. **Text Vectorization:** Vocabulary creation, integer mapping, padding/truncation for uniform sequence length.  
3. **Embedding:** Trainable word embeddings using PyTorch `nn.Embedding`.  
4. **LSTM Layer:** Captures sequential dependencies using forget, input, and output gates.  
5. **Attention Layer:** Highlights important words in a sequence, creating a context vector for classification.  
6. **Classification Layer:** Linear + Softmax layer outputs the predicted sentiment label.




_Model Architecture_           
<img width="380" height="240" alt="Screenshot (83)" src="https://github.com/bishal-pandey/Sentiment-Analysis-on-Product-review/blob/master/Project_Image/architecture.png?raw=true" />
                             
_Attention_         
<img width="380" height="240" alt="Screenshot (83)" src="https://github.com/bishal-pandey/Sentiment-Analysis-on-Product-review/blob/master/Project_Image/attention.jpg" />
                            





