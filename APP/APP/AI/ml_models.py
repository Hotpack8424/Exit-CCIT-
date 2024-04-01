from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

def train_and_evaluate(X_train, X_test, y_train, y_test):
    rf = make_pipeline(StandardScaler(with_mean=False), RandomForestClassifier(n_estimators=200, random_state=42))
    lr = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(C=1, random_state=42))
    svm = make_pipeline(StandardScaler(with_mean=False), SVC(C=1, kernel='rbf', probability=True, random_state=42))
    knn = make_pipeline(StandardScaler(with_mean=False), KNeighborsClassifier(n_neighbors=5))
    xgb = make_pipeline(StandardScaler(with_mean=False), XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=100, max_depth=3, random_state=42))

    base_models = [('rf', rf), ('lr', lr), ('svm', svm), ('knn', knn), ('xgb', xgb)]
    stack_model = StackingClassifier(estimators=base_models, final_estimator=LogisticRegression(), cv=5)

    stack_model.fit(X_train, y_train)

    y_pred_stack = stack_model.predict(X_test)
    accuracy_stack = accuracy_score(y_test, y_pred_stack)

    return stack_model, accuracy_stack
