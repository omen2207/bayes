import csv
import math

def read_data_from_csv(csv_file):
    data = []
    with open(csv_file, mode='r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            data.append(row)
    return data

def calculate_prior_probabilities(data):
    total_samples = len(data)
    class_counts = {}
    for row in data:
        label = row[-1]
        class_counts[label] = class_counts.get(label, 0) + 1
    prior_probabilities = {cls: count / total_samples for cls, count in class_counts.items()}
    return prior_probabilities

def calculate_mean_variance(data, class_counts, numerical_features):
    feature_stats = {}
    for cls in class_counts:
        feature_stats[cls] = []
    for row in data:
        features = [float(row[i]) if i in numerical_features else row[i] for i in range(len(row) - 1)]
        label = row[-1]
        feature_stats[label].append(features)
    for cls in feature_stats:
        features = feature_stats[cls]
        means = []
        variances = []
        for i in range(len(features[0])):
            if isinstance(features[0][i], float):
                mean = sum(f[i] for f in features) / len(features)
                variance = sum((f[i] - mean) ** 2 for f in features) / len(features)
                means.append(mean)
                variances.append(variance)
            else:
                means.append(None)
                variances.append(None)
        feature_stats[cls] = (means, variances)
    return feature_stats

def gaussian_probability(x, mean, variance):
    if variance == 0:
        return 0
    exponent = math.exp(-((x - mean) ** 2) / (2 * variance))
    return (1 / (math.sqrt(2 * math.pi * variance))) * exponent

def calculate_categorical_probabilities(data, class_counts, categorical_features):
    categorical_counts = {}
    for cls in class_counts:
        categorical_counts[cls] = {feature: {} for feature in categorical_features}
    for row in data:
        label = row[-1]
        for feature_index in categorical_features:
            feature_value = row[feature_index]
            if feature_value not in categorical_counts[label][feature_index]:
                categorical_counts[label][feature_index][feature_value] = 0
            categorical_counts[label][feature_index][feature_value] += 1
    return categorical_counts

def predict(test_sample, prior_probabilities, feature_stats, categorical_counts, categorical_features):
    probabilities = {}
    for cls in prior_probabilities:
        prob = prior_probabilities[cls]
        for i in range(len(test_sample)):
            if i in categorical_features:
                feature_value = test_sample[i]
                feature_prob = (categorical_counts[cls][i].get(feature_value, 0) + 1) / (sum(categorical_counts[cls][i].values()) + len(categorical_counts[cls][i]))
                prob *= feature_prob
            else:
                mean, variance = feature_stats[cls][0][i], feature_stats[cls][1][i]
                feature_prob = gaussian_probability(float(test_sample[i]), mean, variance)
                prob *= feature_prob
        
        probabilities[cls] = prob
    
    total_prob = sum(probabilities.values())
    for cls in probabilities:
        probabilities[cls] /= total_prob  
    
    best_class = max(probabilities, key=probabilities.get)
    return best_class, probabilities

csv_file = 'C:\\Users\\bhilw\\OneDrive\\Documents\\DM\\playdata.csv'
data = read_data_from_csv(csv_file)
prior_probabilities = calculate_prior_probabilities(data)

numerical_features = [1, 2]  
categorical_features = [0, 3]  
feature_stats = calculate_mean_variance(data, prior_probabilities, numerical_features)
categorical_counts = calculate_categorical_probabilities(data, prior_probabilities, categorical_features)

outlook = input("Enter Outlook (sunny, overcast, rain): ")
temperature = float(input("Enter Temperature (e.g., 75.0): "))
humidity = float(input("Enter Humidity (e.g., 70.0): "))
windy = input("Is it Windy? (TRUE/FALSE): ")

test_sample = [outlook, temperature, humidity, windy]
predicted_class, probabilities = predict(test_sample, prior_probabilities, feature_stats, categorical_counts, categorical_features)

print(f"Test sample {test_sample} is predicted as: {predicted_class}")
print("Probabilities for each class:")
for cls, prob in probabilities.items():
    print(f"{cls}: {prob:.4f}")