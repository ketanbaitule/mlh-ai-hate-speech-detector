"""
    This File Creates the model and save DecisionTreeClassifier into "classifier.pkl" and CountVectorizer into "vectorizer.pkl"
    Use predict(text) -> [0, 1, 2] 0: Hate Speech, 1: Offensive Speech, 2: Neither of them
"""
import joblib
def predict(text):
    cv = joblib.load('vectorizer.pkl')
    model = joblib.load('classifier.pkl')
    result = cv.transform([text]).toarray()
    return model.predict(result)

# Code to execute if the script is run from the CLI
if __name__ == '__main__':
    import sys
    text = sys.argv[1] if len(sys.argv) >= 1 else ""
    result = predict(text)
    if(result == 0):
        print("Hate Speach Detected")
    elif result == 1:
        print("Offensive Speech Detected")
    else:
        print("No Hate or Offensive Speech Detected")