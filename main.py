#Raseel Jafar 1220724
#Yasmin Al Shawawrh 1220848
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

class ImageClassificationComparator:
    def __init__(self):
        self.models = {}
        self.results = {}
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.class_names = None
        
    def load_dataset(self, dataset_type='fashion-mnist', n_classes=3, samples_per_class=200):
        """
        Load and preprocess dataset
        """
        print(f"Loading {dataset_type} dataset...")
        
        if dataset_type == 'fashion-mnist':
            # Load Fashion-MNIST dataset
            fashion_mnist = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
            X, y = fashion_mnist.data, fashion_mnist.target.astype(int)
            
            # Class names for Fashion-MNIST
            class_names_all = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            
            # Select subset of classes
            selected_classes = list(range(n_classes))
            self.class_names = [class_names_all[i] for i in selected_classes]
            
        elif dataset_type == 'digits':
            # Alternative: Use digits dataset (8x8 images)
            from sklearn.datasets import load_digits
            digits = load_digits()
            X, y = digits.data, digits.target
            selected_classes = list(range(n_classes))
            self.class_names = [f'Digit {i}' for i in selected_classes]
            
        # Filter data for selected classes
        mask = np.isin(y, selected_classes)
        X, y = X[mask], y[mask]
        
        # Sample data to meet requirements (minimum 500 samples)
        total_samples = min(len(X), samples_per_class * n_classes)
        if total_samples < 500:
            samples_per_class = max(500 // n_classes, samples_per_class)
            total_samples = samples_per_class * n_classes
            
        # Balance classes
        X_balanced, y_balanced = [], []
        for class_id in selected_classes:
            class_mask = (y == class_id)
            class_data = X[class_mask]
            class_labels = y[class_mask]
            
            # Sample from this class
            n_samples = min(len(class_data), samples_per_class)
            indices = np.random.choice(len(class_data), n_samples, replace=False)
            
            X_balanced.append(class_data[indices])
            y_balanced.append(class_labels[indices])
            
        X = np.vstack(X_balanced)
        y = np.hstack(y_balanced)
        
        # Normalize pixel values to [0, 1]
        X = X / 255.0
        
        print(f"Dataset loaded: {len(X)} images, {len(np.unique(y))} classes")
        print(f"Image shape: {X.shape[1:]} (flattened)")
        print(f"Classes: {self.class_names}")
        
        return X, y
    
    def preprocess_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split and preprocess the data
        """
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Standardize features for neural network
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        
    def train_naive_bayes(self):
        """
        Train Naive Bayes classifier
        """
        print("\nTraining Naive Bayes Classifier...")
        start_time = time.time()
        
        # Use Gaussian Naive Bayes
        nb_model = GaussianNB()
        nb_model.fit(self.X_train, self.y_train)
        
        training_time = time.time() - start_time
        
        # Predictions
        y_pred = nb_model.predict(self.X_test)
        
        self.models['Naive Bayes'] = nb_model
        self.results['Naive Bayes'] = {
            'predictions': y_pred,
            'training_time': training_time
        }
        
        print(f"Naive Bayes training completed in {training_time:.2f} seconds")
        
    def train_decision_tree(self):
        """
        Train Decision Tree classifier
        """
        print("\nTraining Decision Tree Classifier...")
        start_time = time.time()
        
        # Decision Tree with limited depth to prevent overfitting
        dt_model = DecisionTreeClassifier(
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42
        )
        dt_model.fit(self.X_train, self.y_train)
        
        training_time = time.time() - start_time
        
        # Predictions
        y_pred = dt_model.predict(self.X_test)
        
        self.models['Decision Tree'] = dt_model
        self.results['Decision Tree'] = {
            'predictions': y_pred,
            'training_time': training_time
        }
        
        print(f"Decision Tree training completed in {training_time:.2f} seconds")
        
    def train_neural_network(self):
        """
        Train Multi-Layer Perceptron (Neural Network)
        """
        print("\nTraining Neural Network (MLP)...")
        start_time = time.time()
        
        # MLP with 2 hidden layers
        mlp_model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10
        )
        mlp_model.fit(self.X_train_scaled, self.y_train)
        
        training_time = time.time() - start_time
        
        # Predictions
        y_pred = mlp_model.predict(self.X_test_scaled)
        
        self.models['Neural Network'] = mlp_model
        self.results['Neural Network'] = {
            'predictions': y_pred,
            'training_time': training_time
        }
        
        print(f"Neural Network training completed in {training_time:.2f} seconds")
        
    def evaluate_models(self):
        """
        Evaluate all models and compute metrics
        """
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        metrics_summary = []
        
        for model_name in self.models.keys():
            y_pred = self.results[model_name]['predictions']
            training_time = self.results[model_name]['training_time']
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            metrics_summary.append({
                'Model': model_name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1-Score': f1,
                'Training Time (s)': training_time
            })
            
            print(f"\n{model_name}:")
            print(f"  Accuracy:  {accuracy:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall:    {recall:.4f}")
            print(f"  F1-Score:  {f1:.4f}")
            print(f"  Training Time: {training_time:.2f}s")
            
        return metrics_summary
    
    def plot_confusion_matrices(self):
        """
        Plot confusion matrices for all models
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, model_name in enumerate(self.models.keys()):
            y_pred = self.results[model_name]['predictions']
            cm = confusion_matrix(self.y_test, y_pred)
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=self.class_names, 
                       yticklabels=self.class_names,
                       ax=axes[idx])
            axes[idx].set_title(f'{model_name}\nConfusion Matrix')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
            
        plt.tight_layout()
        plt.show()
        
    def plot_performance_comparison(self, metrics_summary):
        """
        Plot performance comparison
        """
        import pandas as pd
        
        df = pd.DataFrame(metrics_summary)
        
        # Performance metrics plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy, Precision, Recall, F1-Score
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(df))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            ax1.bar(x + i*width, df[metric], width, label=metric, alpha=0.8)
            
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Score')
        ax1.set_title('Model Performance Comparison')
        ax1.set_xticks(x + width * 1.5)
        ax1.set_xticklabels(df['Model'])
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # Training time comparison
        ax2.bar(df['Model'], df['Training Time (s)'], color='orange', alpha=0.7)
        ax2.set_xlabel('Models')
        ax2.set_ylabel('Training Time (seconds)')
        ax2.set_title('Training Time Comparison')
        
        plt.tight_layout()
        plt.show()
        
    def visualize_sample_predictions(self, n_samples=9):
        """
        Visualize sample predictions from test set
        """
        # Get sample indices
        sample_indices = np.random.choice(len(self.X_test), n_samples, replace=False)
        
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        axes = axes.ravel()
        
        for i, idx in enumerate(sample_indices):
            # Reshape image for visualization (assuming 28x28 for Fashion-MNIST)
            img_size = int(np.sqrt(self.X_test.shape[1]))
            image = self.X_test[idx].reshape(img_size, img_size)
            
            axes[i].imshow(image, cmap='gray')
            axes[i].set_title(f'True: {self.class_names[self.y_test[idx]]}')
            axes[i].axis('off')
            
            # Add predictions from all models
            pred_text = ""
            for model_name in self.models.keys():
                y_pred = self.results[model_name]['predictions']
                pred_class = self.class_names[y_pred[idx]]
                pred_text += f"{model_name}: {pred_class}\n"
            
            axes[i].text(0, -0.1, pred_text, transform=axes[i].transAxes, 
                        fontsize=8, verticalalignment='top')
        
        plt.tight_layout()
        plt.show()
        
    def generate_detailed_report(self, metrics_summary):
        """
        Generate detailed classification report
        """
        print("\n" + "="*60)
        print("DETAILED CLASSIFICATION REPORT")
        print("="*60)
        
        for model_name in self.models.keys():
            y_pred = self.results[model_name]['predictions']
            
            print(f"\n{model_name}:")
            print("-" * 30)
            print(classification_report(self.y_test, y_pred, 
                                      target_names=self.class_names))

def main():
    """
    Main function to run the complete image classification comparison
    """
    # Initialize the comparator
    comparator = ImageClassificationComparator()
    
    # Load dataset
    print("Image Classification Comparison Project")
    print("="*50)
    
    # You can change dataset_type to 'digits' for alternative dataset
    X, y = comparator.load_dataset(dataset_type='fashion-mnist', 
                                  n_classes=5, 
                                  samples_per_class=800)
    
    # Preprocess data
    comparator.preprocess_data(X, y)
    
    # Train all models
    comparator.train_naive_bayes()
    comparator.train_decision_tree()
    comparator.train_neural_network()
    
    # Evaluate models
    metrics_summary = comparator.evaluate_models()
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    comparator.plot_confusion_matrices()
    comparator.plot_performance_comparison(metrics_summary)
    comparator.visualize_sample_predictions()
    
    # Generate detailed report
    comparator.generate_detailed_report(metrics_summary)
    
    print("\n" + "="*50)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*50)
    
    return comparator, metrics_summary

if __name__ == "__main__":
    # Run the complete project
    comparator, results = main()