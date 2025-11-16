import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from Task1 import execute, Features, classes, preprocessing_data, visualize_features,plot_decision_boundary

data=preprocessing_data()

def user_inputs():
    first_feature = feature1_var.get()
    second_feature = feature2_var.get()

    if first_feature == second_feature:
        messagebox.showerror("Invalid Features","You cannot select the same feature twice. Please choose two different features.")
        return

    first_class = class1_var.get()
    second_class = class2_var.get()

    if first_class == second_class:
        messagebox.showerror("Invalid Classes","You cannot select the same class twice. Please choose two different classes.")
        return

    try:
        learning_rate = float(learning_rate_var.get())
        if learning_rate <= 0 :
            messagebox.showerror("Invalid Input","Learning rate must be a positive number greater than 0.")
            return
    except ValueError:
        messagebox.showerror("Invalid Input","Learning rate must be a numeric value.")
        return

    try:
        epochs = int(epochs_var.get())
        if epochs <= 0:
            messagebox.showerror("Invalid Input","Number of epochs must be a positive number greater than 0.")
            return
    except ValueError:
        messagebox.showerror("Invalid Input","Number of epochs must be an integer value.")
        return

    algorithm = algorithm_var.get()
    if algorithm == "Adaline":
        try:
            mse_threshold = float(mse_var.get())
            if mse_threshold <= 0:
                messagebox.showerror("Invalid Input","MSE threshold must be a positive number grater than 0.")
                return
        except ValueError:
            messagebox.showerror("Invalid Input","MSE threshold must be a numeric value.")
            return
    else:
        mse_threshold = None
    bias = bias_var.get()
    results=execute(first_feature, second_feature, first_class, second_class,learning_rate,epochs,bias,algorithm,mse_threshold,data)
    display_results(first_feature, second_feature, first_class, second_class,algorithm,results)


def display_results(first_feature, second_feature, first_class, second_class, algorithm,results):
    result_label_text = (
        f"Train Accuracy: {results['train_overall_accuracy']:.2f}%\n\n"
        f"Train Confusion Matrix:\n{results['train_confusion_matrix']}\n\n"
        f"Test Accuracy: {results['test_overall_accuracy']:.2f}%\n\n"
        f"Test Confusion Matrix:\n{results['test_confusion_matrix']}"
    )

    if algorithm == "Adaline":
        result_label_text += f"\n\nTrain MSE: {results['mse_train']:.4f}\nTest MSE: {results['mse_test']:.4f}"

    result_label.config(text=result_label_text)
    plot_decision_boundary(
        results['test_features'],
        results['test_target'],
        first_class,
        second_class,
        first_feature,
        second_feature,
        results['final_weights'],
        results['final_bias']
    )

def update_mse_visibility():
    if algorithm_var.get() == "Adaline":
        mse_label.grid()
        mse_entry.grid()
    else:
        mse_label.grid_remove()
        mse_entry.grid_remove()


def visualization():
    visualize_features(data)

root = tk.Tk()
root.title("Penguin Classifier GUI")
root.geometry("750x750")

feature_frame = ttk.LabelFrame(root, text="Select Features", padding=10)
feature_frame.grid(row=0, column=0, padx=20, pady=10, sticky="ew")

ttk.Label(feature_frame, text="Feature 1:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
feature1_var = tk.StringVar(value=Features[0])
feature1_menu = ttk.Combobox(feature_frame, textvariable=feature1_var, values=Features, state="readonly")
feature1_menu.grid(row=0, column=1, padx=5, pady=5)

ttk.Label(feature_frame, text="Feature 2:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
feature2_var = tk.StringVar(value=Features[1])
feature2_menu = ttk.Combobox(feature_frame, textvariable=feature2_var, values=Features, state="readonly")
feature2_menu.grid(row=1, column=1, padx=5, pady=5)

class_frame = ttk.LabelFrame(root, text="Select Classes", padding=10)
class_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")

ttk.Label(class_frame, text="Class 1:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
class1_var = tk.StringVar(value=classes[0])
class1_menu = ttk.Combobox(class_frame, textvariable=class1_var, values=classes, state="readonly")
class1_menu.grid(row=0, column=1, padx=5, pady=5)

ttk.Label(class_frame, text="Class 2:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
class2_var = tk.StringVar(value=classes[1])
class2_menu = ttk.Combobox(class_frame, textvariable=class2_var, values=classes, state="readonly")
class2_menu.grid(row=1, column=1, padx=5, pady=5)

parameters_frame = ttk.LabelFrame(root, text="Parameters", padding=10)
parameters_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")

ttk.Label(parameters_frame, text="Learning rate (eta):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
learning_rate_var = tk.StringVar(value="0.01")
ttk.Entry(parameters_frame, textvariable=learning_rate_var).grid(row=0, column=1, padx=5, pady=5)

ttk.Label(parameters_frame, text="Number of epochs:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
epochs_var = tk.StringVar(value="50")
ttk.Entry(parameters_frame, textvariable=epochs_var).grid(row=1, column=1, padx=5, pady=5)

bias_var = tk.BooleanVar(value=True)
ttk.Checkbutton(parameters_frame, text="Add bias", variable=bias_var).grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=5)

mse_label = ttk.Label(parameters_frame, text="MSE threshold:")
mse_entry = ttk.Entry(parameters_frame)
mse_var = tk.StringVar(value="0.001")
mse_entry.config(textvariable=mse_var)
mse_label.grid(row=3, column=0, sticky="w", padx=5, pady=5)
mse_entry.grid(row=3, column=1, padx=5, pady=5)

mse_label.grid_remove()
mse_entry.grid_remove()

algorithm_frame = ttk.LabelFrame(root, text="Algorithm", padding=10)
algorithm_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")

algorithm_var = tk.StringVar(value="Perceptron")

ttk.Radiobutton(algorithm_frame, text="Perceptron", variable=algorithm_var, value="Perceptron", command=update_mse_visibility).grid(row=0, column=0, padx=5, pady=5)
ttk.Radiobutton(algorithm_frame, text="Adaline", variable=algorithm_var, value="Adaline", command=update_mse_visibility).grid(row=0, column=1, padx=5, pady=5)

btn_frame = ttk.Frame(root, padding=10)
btn_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")

run_button = ttk.Button(btn_frame, text="Run", command=user_inputs)
run_button.grid(row=0, column=0, padx=10, pady=5)

visualize_button = ttk.Button(btn_frame, text="Features Visualizations",command=visualization)
visualize_button.grid(row=0, column=1, padx=10, pady=5)

result_frame = ttk.LabelFrame(root, text="Results", padding=10)
result_frame.grid(row=0, column=1, padx=20, pady=10, sticky="nsew")

result_label = ttk.Label(result_frame, text="Results will appear here", justify="left", anchor="nw")
result_label.pack(fill="both", expand=True)

root.mainloop()
