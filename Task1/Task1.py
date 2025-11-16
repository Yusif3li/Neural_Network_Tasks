import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


Features = ["CulmenLength", "CulmenDepth", "FlipperLength", "OriginLocation", "BodyMass"]
classes = ["Adelie", "Chinstrap", "Gentoo"]

def preprocessing_data():
    data = pd.read_csv("penguins.csv")
    target=data["Species"]
    data = data.drop("Species", axis=1)
    # print(data.info)
    # print(data.isnull().sum())
    # print(data.dtypes)
    # print(data.duplicated().sum())

    numeric_data = data.select_dtypes(include=["number"])
    categorical_data = data.drop(columns=numeric_data.columns)

    numeric_data=numeric_data.fillna(numeric_data.mean())
    categorical_data=categorical_data.fillna(categorical_data.mode().iloc[0])

    data=pd.concat([numeric_data,categorical_data], axis=1)

    encoder=LabelEncoder()
    data["OriginLocation"] = encoder.fit_transform(data["OriginLocation"])
    data["Species"] = target
    return data

def visualize_features(data):
    all_combinations = []
    for i in range(len(Features)):
        for j in range(i + 1,len(Features)):
            all_combinations.append((Features[i],Features[j]))
    rows=3
    columns=4
    index=1
    plt.figure(figsize=(18, 8))
    for feature1,feature2 in all_combinations:
        plt.subplot(rows, columns, index)
        sns.scatterplot(data=data, x=feature1, y=feature2, hue="Species", legend='full')
        plt.legend(fontsize=6, loc='best')
        plt.title(f"{feature1} vs {feature2}")
        plt.xlabel(feature1)
        plt.ylabel(feature2)
        index += 1
    plt.tight_layout(pad=2.0)
    plt.show()

def signum(value):
    if value>=0:
        y=1
    else:
        y=-1
    return y

def perceptron_train(features,target,weights,bias,epochs,learning_rate,use_bias):
    num_of_errors =0
    for epoch in range(epochs):
        for x in range(len(features)):
            result= np.dot(weights.T,features[x])+bias
            y=signum(result)
            if y != target[x]:
                num_of_errors+=1
                loss = target[x]-y
                weights=weights+(loss*learning_rate*features[x])
                if use_bias:
                 bias = bias + (loss*learning_rate)
        if num_of_errors == 0:
            return weights, bias
        num_of_errors=0
    return weights,bias

def perceptron_test(features,target,weights,bias):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for x in range(len(features)):
        result = np.dot(weights.T, features[x]) + bias
        y=signum(result)
        if y == 1 and target[x]==1:
            tp+=1
        elif y ==-1 and target[x]==-1:
            tn +=1
        elif y ==1 and target[x]==-1:
            fp+=1
        elif y ==-1 and target[x]==1:
            fn+=1
    confusion_matrix= np.array([[tp,fn],[fp,tn]])
    overall_accuracy = ((tp + tn ) / len(features))*100
    return confusion_matrix,overall_accuracy

def adaline_train(features,target,weights,mse,epochs,learning_rate, bias,use_bias):
    for epoch in range(epochs):
        sum_err = 0
        for x in range(len(features)):
            res = np.dot(weights.T, features[x]) + bias
            error = target[x] - res
            weights = weights + (learning_rate * error * features[x])
            if use_bias:
             bias = bias + (learning_rate * error)
        for x in range(len(features)):
            res = np.dot(weights.T, features[x]) + bias
            error = target[x] - res
            sum_err = sum_err + pow(error, 2)
        mse_train = (sum_err / features.shape[0]) / 2
        if mse_train <= mse:
            return weights,bias
    return weights,bias


def adaline_test(features,target,weights, bias):
        tp, tn, fp, fn,sum_err = 0, 0, 0, 0,0
        for x in range(len(features)):
            res = np.dot(weights.T,features[x]) + bias
            res=signum(res)
            error = target[x] - res
            sum_err = sum_err + pow(error, 2)
            if target[x] == 1:
                if res == 1:
                    tp += 1
                else:
                    fn += 1
            if target[x] == -1:
                if res == -1:
                    tn += 1
                else:
                    fp += 1
        mse_train = (sum_err / features.shape[0]) / 2
        acc = ((tp + tn) / (tp + tn + fp + fn))*100
        confusion_matrix = np.array([[tp, fn], [fp, tn]])
        return confusion_matrix, acc,mse_train


def plot_decision_boundary(features,target,c1,c2 ,f1, f2, w, bias):
    features = pd.DataFrame(features, columns=[f1, f2])
    target = pd.DataFrame({'Species':target})

    result = pd.concat([features, target], axis=1)
    class1 = result[result['Species'] == 1]
    class2 = result[result['Species'] == -1]

    plt.close("all")
    plt.scatter(class1[f1], class1[f2], color='blue', label= c1)
    plt.scatter(class2[f1], class2[f2], color='red', label = c2)

    w1, w2 = w[0], w[1]

    if w2==0:
     x1 = np.array([-bias / w1, -bias / w1])
     x2 = np.array([features[f2].min(), features[f2].max()])
    else:
     x1 = np.array([features[f1].min(), features[f1].max()])
     x2 = -(w1 * x1 + bias) / w2

    plt.plot(x1, x2, color='green', linewidth=2, label='Decision Boundary')


    plt.xlabel(f1)
    plt.ylabel(f2)
    plt.legend()
    plt.title(" Decision Boundary")
    plt.grid(True)
    plt.show()


def execute(first_feature, second_feature, first_class, second_class, learning_rate, epochs, use_bias, algorithm,mse_threshold,preprocessed_data):

    class1 = preprocessed_data[preprocessed_data['Species'].isin([first_class])].copy()
    class1['Species'] = 1

    class2 = preprocessed_data[preprocessed_data['Species'].isin([second_class])].copy()
    class2['Species'] = -1


    class1_features = class1[[first_feature, second_feature]]
    class2_features = class2[[first_feature, second_feature]]

    class1_target = class1['Species']
    class2_target = class2['Species']

    class1_features_train, class1_features_test, class1_target_train, class1_target_test = train_test_split(
        class1_features, class1_target, test_size=0.4, shuffle=True, random_state=42)
    class2_features_train, class2_features_test, class2_target_train, class2_target_test = train_test_split(
        class2_features, class2_target, test_size=0.4, shuffle=True, random_state=42)

    train_features = np.array(pd.concat([class1_features_train, class2_features_train], axis=0))
    test_features = np.array(pd.concat([class1_features_test, class2_features_test], axis=0))
    train_target = np.array(pd.concat([class1_target_train, class2_target_train], axis=0))
    test_target = np.array(pd.concat([class1_target_test, class2_target_test], axis=0))
    train_features, train_target = shuffle(train_features, train_target, random_state=42)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)
    test_features = scaler.transform(test_features)

    np.random.seed(42)

    weights = np.random.rand(2)
    if use_bias:
        bias = np.random.rand()
    else:
        bias = 0
    if algorithm == 'Perceptron':
        final_weights, final_bias = perceptron_train(train_features, train_target, weights, bias, epochs, learning_rate,use_bias)
        train_confusion_matrix,train_overall_accuracy=perceptron_test(train_features,train_target, final_weights,final_bias)
        test_confusion_matrix,test_overall_accuracy=perceptron_test(test_features,test_target, final_weights,final_bias)
        results = {
            'test_features': test_features,
            'test_target': test_target,
            'final_weights': final_weights,
            'final_bias': final_bias,
            'train_confusion_matrix': train_confusion_matrix,
            'train_overall_accuracy': train_overall_accuracy,
            'test_confusion_matrix': test_confusion_matrix,
            'test_overall_accuracy': test_overall_accuracy
        }
    else:
        final_weights, final_bias = adaline_train(train_features,train_target,weights,mse_threshold,epochs,learning_rate, bias,use_bias)
        train_confusion_matrix,train_overall_accuracy,mse_train=adaline_test(train_features,train_target, final_weights, final_bias)
        test_confusion_matrix,test_overall_accuracy,mse_test=adaline_test(test_features,test_target, final_weights,final_bias)
        results = {
            'test_features': test_features,
            'test_target': test_target,
            'final_weights': final_weights,
            'final_bias': final_bias,
            'train_confusion_matrix': train_confusion_matrix,
            'train_overall_accuracy': train_overall_accuracy,
            'test_confusion_matrix': test_confusion_matrix,
            'test_overall_accuracy': test_overall_accuracy,
            'mse_train': mse_train,
            'mse_test': mse_test
        }
    return results



