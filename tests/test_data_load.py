from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import fetch_20newsgroups

# 加载数据
data = fetch_20newsgroups(subset='all', categories=['sci.space', 'rec.sport.hockey'], remove=('headers', 'footers', 'quotes'))

X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义Pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('classifier', LogisticRegression(max_iter=1000))
])

# 自定义参数设置
# 使用`Pipeline`时参数以<component_name>__<param_name>格式传递
pipeline.set_params(tfidf__max_features=5000, classifier__C=1.0)

# 拟合Pipeline
pipeline.fit(X_train, y_train)

# 测试模型
accuracy = pipeline.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")
