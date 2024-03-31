from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def preprocess_and_vectorize(df):
    df.fillna('', inplace=True)
    X = df['url'] + " " + df['connect'] + " " + df['meta'] + " " + df['a'] + " " + df['title'] + " " + df['tld'] + " " + df['https'] + " " + df['ip']
    y = df['label']

    vectorizer = TfidfVectorizer()
    X_vectorized = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, vectorizer
