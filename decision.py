
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from matplotlib.colors import ListedColormap

st.image(r"new logo.jpg")
# Set the title of the app
st.title("Decision Surface Visualization")

# Create a sidebar
st.sidebar.header("User's Input")

# Classification algorithm selection
classifier_name = st.sidebar.selectbox(
    "Select Classifier",
    ("KNN", "SVM", "Decision Tree", "Random Forest","Logistic Regression")
)

# Add a select box to the sidebar
dataset_name = st.sidebar.selectbox(
    "Select Dataset",
    ("Moons", "Circles", "Blobs", "U-Shaped", "Two-Spirals", "Overlap", "XOR")
)

n_samples=300

# Add sliders and number inputs to the sidebar based on the selected dataset
if dataset_name in ("Moons", "Circles", "Two-Spirals", "Overlap", "XOR"):
    noise = st.sidebar.slider("Noise Level", 0.0, 1.0, 0.3, 0.1)
if dataset_name == "Circles":
    factor = st.sidebar.slider("Factor", 0.1, 1.0, 0.5, 0.1)
if dataset_name == "Blobs":
    noise = st.sidebar.slider("Cluster Standard Deviation", 0.0, 5.0, 1.0, 0.1)
    centers = st.sidebar.number_input("Number of Centers", 1, 5, 3)


# KNN Classifier parameters
if classifier_name == "KNN":
    n_neighbors = st.sidebar.slider("Number of Neighbors(K-Value)", 1, 15, 5)
    metric = st.sidebar.selectbox("Distance Metric", ("euclidean", "manhattan", "minkowski"))
    algorithm = st.sidebar.selectbox("Algorithm(type)",("auto","kd_tree","ball_tree","brute"))
# LogisticRegression parameters
if classifier_name == "Logistic Regression":
    multi_class = st.sidebar.selectbox("Multi-class", ["auto", "ovr", "multinomial"])
    max_iter = st.sidebar.slider("Max Iterations", 50, 500, 100)

# SVM Classifier parameters
if classifier_name == "SVM":
    C = st.sidebar.slider("C (Regularization parameter)", 0.01, 10.0, 1.0)
    kernel = st.sidebar.selectbox("Kernel", ("linear", "poly", "rbf", "sigmoid"))

# Decision Tree Classifier parameters
if classifier_name == "Decision Tree":
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)

# Random Forest Classifier parameters
if classifier_name == "Random Forest":
    n_estimators = st.sidebar.slider("Number of Trees", 10, 100, 50)
    max_depth = st.sidebar.slider("Max Depth", 1, 20, 5)

# Generate dataset based on user selection
def generate_u_shape(n_samples):
    np.random.seed(0)
    X = np.random.rand(n_samples, 2)
    y = (X[:, 1] > np.abs(X[:, 0] - 0.5)).astype(int)
    return X, y

def generate_two_spirals(n_points, noise=0.5):
    n = np.sqrt(np.random.rand(n_points, 1)) * 780 * (2 * np.pi) / 360
    d1x = -np.cos(n) * n + np.random.rand(n_points, 1) * noise
    d1y = np.sin(n) * n + np.random.rand(n_points, 1) * noise
    X = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y))))
    y = np.hstack((np.zeros(n_points), np.ones(n_points)))
    return X, y

def generate_overlap(n_samples):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] + X[:, 1] > 1).astype(int)
    return X, y

def generate_xor(n_samples):
    np.random.seed(0)
    X = np.random.rand(n_samples, 2) * 2 - 1
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    return X, y

if dataset_name == "U-Shaped":
    X, y = generate_u_shape(n_samples)
elif dataset_name == "Two-Spirals":
    X, y = generate_two_spirals(n_samples, noise=noise)
elif dataset_name == "Overlap":
    X, y = generate_overlap(n_samples)
elif dataset_name == "XOR":
    X, y = generate_xor(n_samples)
elif dataset_name == "Moons":
    X, y = make_moons(noise=noise, random_state=0)
elif dataset_name == "Circles":
    X, y = make_circles(noise=noise, factor=factor, random_state=1)
elif dataset_name == "Blobs":
    X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=0, cluster_std=noise)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=st.sidebar.number_input("Random State", 0, 100, 42, 1))

# Train the selected classifier
if classifier_name == "KNN":
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric,algorithm=algorithm)
elif classifier_name == "SVM":
    classifier = SVC(C=C, kernel=kernel, random_state=42)
elif classifier_name == "Decision Tree":
    classifier = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
elif classifier_name == "Random Forest":
    classifier = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
elif classifier_name == "Logistic Regression":
    classifier = LogisticRegression(multi_class=multi_class,max_iter=max_iter)

classifier.fit(X_train, y_train)

# Make predictions
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Plot the decision surface
h = .02  # step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = classifier.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)



plt.figure(figsize=(6, 4))
plt.contourf(xx, yy, Z, alpha=0.8,cmap = ListedColormap(["yellow","red"]))
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o',cmap=ListedColormap(["orange","yellow"]))
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title(f"Decision Surface of {dataset_name} Dataset\nAccuracy: {accuracy:.2f}")

# Display the plot in Streamlit
st.pyplot(plt)
