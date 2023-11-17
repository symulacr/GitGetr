import numpy as np
from sklearn.preprocessing import StandardScaler

def normalize_data(data):
    """
    Normalize the input data between 0 and 1.
    """
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

def remove_outliers(data):
    """
    Remove outliers from the input data using z-score method.
    """
    z_scores = np.abs((data - np.mean(data)) / np.std(data))
    threshold = 3
    data_no_outliers = data[z_scores < threshold]
    return data_no_outliers

def impute_missing_values(data):
    """
    Impute missing values in the data by replacing them with the mean.
    """
    nan_indices = np.isnan(data)
    mean_val = np.nanmean(data)
    data[nan_indices] = mean_val
    return data

def scale_features(data):
    """
    Scale features using StandardScaler.
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data

def encode_categorical_variables(data):
    """
    Encode categorical variables using one-hot encoding.
    """
    # Example code for categorical encoding (adjust as per data types)
    encoded_data = pd.get_dummies(data)
    return encoded_data

def feature_engineering(data):
    """
    Perform feature engineering on the input data.
    """
    # Example: Adding a new feature as a transformation of existing features
    data['new_feature'] = data['feature1'] * data['feature2']
    return data

def preprocess_data(data):
    """
    Perform complete preprocessing pipeline on the input data.
    """
    # Example: Apply a series of preprocessing steps
    processed_data = remove_outliers(data)
    processed_data = impute_missing_values(processed_data)
    processed_data = scale_features(processed_data)
    processed_data = feature_engineering(processed_data)
    return processed_data
import numpy as np

# Function 1
def normalize_data(data):
    # Normalize data between 0 and 1
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    return normalized_data

# Function 2
def standardize_data(data):
    # Standardize data (mean=0, std=1)
    standardized_data = (data - np.mean(data)) / np.std(data)
    return standardized_data

# Function 3
def binarize_data(data, threshold=0.5):
    # Binarize data based on a threshold
    binarized_data = np.where(data >= threshold, 1, 0)
    return binarized_data

# Function 4
def fill_missing_values(data, value=0):
    # Fill missing values in data with a specified value
    filled_data = np.nan_to_num(data, nan=value)
    return filled_data

# Function 5
def remove_outliers(data, threshold=3):
    # Remove outliers beyond a certain threshold
    mean = np.mean(data)
    std_dev = np.std(data)
    filtered_data = data[(data - mean) < threshold * std_dev]
    return filtered_data

# Function 6
def scale_features(data, scaling_factor=10):
    # Scale features by a specified factor
    scaled_data = data * scaling_factor
    return scaled_data

# Function 7
def log_transform(data):
    # Log transform data
    return np.log(data)

# Function 8
def exponential_transform(data):
    # Exponential transform data
    return np.exp(data)

# Function 9
def square_root_transform(data):
    # Square root transform data
    return np.sqrt(data)

# Function 10
def min_max_scaling(data):
    # Perform Min-Max scaling
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Function 11
def z_score_scaling(data):
    # Perform Z-score scaling
    return (data - np.mean(data)) / np.std(data)

# Function 12
def clip_values(data, min_val, max_val):
    # Clip values within a specified range
    return np.clip(data, min_val, max_val)

# Function 13
def remove_duplicates(data):
    # Remove duplicate entries from data
    return np.unique(data)

# Function 14
def shuffle_data(data):
    # Shuffle the order of data entries
    np.random.shuffle(data)
    return data

# Function 15
def extract_features(data, features):
    # Extract specific features from data
    return data[features]

# Function 16
def impute_missing_values(data, strategy='mean'):
    # Impute missing values based on specified strategy
    if strategy == 'mean':
        return np.nanmean(data)
    elif strategy == 'median':
        return np.nanmedian(data)
    elif strategy == 'mode':
        # Calculate mode for each column
        return np.nanargmax(np.bincount(data.astype(int)))

# Function 17
def rescale_data(data, new_min=0, new_max=1):
    # Rescale data to a new range
    return (data - np.min(data)) * (new_max - new_min) / (np.max(data) - np.min(data)) + new_min
