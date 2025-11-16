import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from Task2 import DataProcessor, MLP

class GUI:
    def __init__(self, main_menu):
        self.main_menu = main_menu
        self.main_menu.title("Penguin Classifier (Task 2)")
        self.main_menu.geometry("700x950")

        app_style = ttk.Style()
        app_style.configure("TFrame", background="#f0f0f0")
        app_style.configure("TLabel", background="#f0f0f0", font=("Arial", 11))
        app_style.configure("TButton", font=("Arial", 10, "bold"))
        app_style.configure("TEntry", font=("Arial", 11))
        app_style.configure("TRadiobutton", background="#f0f0f0", font=("Arial", 10))
        app_style.configure("TCheckbutton", background="#f0f0f0")

        content_frame = ttk.Frame(main_menu, padding="20")
        content_frame.pack(fill=tk.BOTH, expand=True)

        guideline_frame = ttk.Labelframe(content_frame, text="User Guidelines", padding="15")
        guideline_frame.pack(fill="x", expand="yes", pady=10)
        
        guideline_text = (
            "1. Number of Hidden Layers: Must be a positive integer (e.g., 2).\n"
            "2. Check 'Same size for all layers' if all layers have the same neuron count.\n"
            "3. If checked: 'Neurons per Layer' must be one positive integer (e.g., 10).\n"
            "4. If unchecked: 'Neurons per Layer' must be a comma-separated list (e.g., 10, 5).\n"
            "5. Input Match: 'Number of Layers' must match the count in 'Neurons per Layer' (if unchecked).\n"
            "6. LR and Epochs: Must be positive numbers."
        )
        ttk.Label(guideline_frame, text=guideline_text, justify=tk.LEFT).pack(anchor="w")

        hyperparameter_frame = ttk.Labelframe(content_frame, text="Hyperparameters Input", padding="15")
        hyperparameter_frame.pack(fill="x", expand="yes", pady=10)
        
        ttk.Label(hyperparameter_frame, text="Number of Hidden Layers:").grid(row=0, column=0, sticky="w", pady=5)
        self.num_hidden_layers_input = ttk.Entry(hyperparameter_frame, width=20)
        self.num_hidden_layers_input.grid(row=0, column=1, pady=5)
        self.num_hidden_layers_input.insert(0, "1")

        ttk.Label(hyperparameter_frame, text="Same size for all layers?:").grid(row=1, column=0, sticky="w", pady=5)
        self.same_neurons_checkbox = tk.BooleanVar(value=True)
        ttk.Checkbutton(hyperparameter_frame, text="", variable=self.same_neurons_checkbox).grid(row=1, column=1, sticky="w")

        ttk.Label(hyperparameter_frame, text="Neurons per Layer (e.g., '10' or '10,5'):").grid(row=2, column=0, sticky="w", pady=5)
        self.hidden_layers_input = ttk.Entry(hyperparameter_frame, width=20)
        self.hidden_layers_input.grid(row=2, column=1, pady=5)
        self.hidden_layers_input.insert(0, "10")
        
        ttk.Label(hyperparameter_frame, text="Learning Rate:").grid(row=3, column=0, sticky="w", pady=5)
        self.learning_rate_input = ttk.Entry(hyperparameter_frame, width=20)
        self.learning_rate_input.grid(row=3, column=1, pady=5)
        self.learning_rate_input.insert(0, "0.01")

        ttk.Label(hyperparameter_frame, text="Epochs:").grid(row=4, column=0, sticky="w", pady=5)
        self.epochs_input = ttk.Entry(hyperparameter_frame, width=20)
        self.epochs_input.grid(row=4, column=1, pady=5)
        self.epochs_input.insert(0, "100")

        ttk.Label(hyperparameter_frame, text="Activation Function:").grid(row=5, column=0, sticky="w", pady=5)
        self.activation_func_choice = tk.StringVar(value="sigmoid")
        ttk.Radiobutton(hyperparameter_frame, text="Sigmoid", variable=self.activation_func_choice, value="sigmoid").grid(row=5, column=1, sticky="w")
        ttk.Radiobutton(hyperparameter_frame, text="Tanh", variable=self.activation_func_choice, value="tanh").grid(row=5, column=1, sticky="e")

        ttk.Label(hyperparameter_frame, text="Apply Bias:").grid(row=6, column=0, sticky="w", pady=5)
        self.use_bias_checkbox = tk.BooleanVar(value=True)
        ttk.Checkbutton(hyperparameter_frame, text="", variable=self.use_bias_checkbox).grid(row=6, column=1,sticky="w")
        
        start_button = ttk.Button(content_frame, text="Start", command=self.Start_Classification)
        start_button.pack(pady=20, fill="x")

        results_display_frame = ttk.Labelframe(content_frame, text="Results", padding="15")
        results_display_frame.pack(fill="both", expand="yes")

        self.test_accuracy_label = ttk.Label(results_display_frame, text="Test Accuracy: N/A", font=("Arial", 12, "bold"))
        self.test_accuracy_label.pack(pady=10)

        self.matrix_plot_frame = ttk.Frame(results_display_frame)
        self.matrix_plot_frame.pack(fill="both", expand=True)
        self.plot_canvas = None

    def Start_Classification(self):
        
        is_same_size = self.same_neurons_checkbox.get()
        try:
            num_layers = int(self.num_hidden_layers_input.get())
            if num_layers < 0:
                messagebox.showerror("Invalid Input", "Number of Hidden Layers must be a non-negative integer.")
                return
        except ValueError:
            messagebox.showerror("Invalid Input", "Number of Hidden Layers must be an integer.")
            return

        neuron_input_str = self.hidden_layers_input.get().strip()
        hidden_layers = []
        if is_same_size:
            try:
                if num_layers == 0:
                     neuron_input_str = "0"
                
                neuron_count = int(neuron_input_str)
                
                if neuron_count <= 0 and num_layers > 0:
                     messagebox.showerror("Invalid Input", "Neuron count must be a positive integer.")
                     return
                elif num_layers > 0:
                    hidden_layers = [neuron_count] * num_layers

            except ValueError:
                messagebox.showerror("Invalid Input", "When 'Same size' is checked, 'Neurons per Layer' must be a single integer (e.g., '10').")
                return
        else:
            hidden_layers_str_list = neuron_input_str.split(',')
            try:
                for n in hidden_layers_str_list:
                    clean_n = n.strip()
                    if clean_n:
                        neuron_count = int(clean_n)
                        if neuron_count <= 0:
                             messagebox.showerror("Invalid Input", "Neuron counts must all be positive integers.")
                             return
                        hidden_layers.append(neuron_count)
            except ValueError:
                messagebox.showerror("Invalid Input", "Neurons per Layer must be a comma-separated list of integers (e.g., '10, 5').")
                return
            
            if len(hidden_layers) != num_layers:
                messagebox.showerror("Input Mismatch",
                    f"Number of layers was set to: {num_layers}\n"
                    f"But neuron list has {len(hidden_layers)} entries: {hidden_layers}\n\n"
                    f"Please ensure the numbers match.")
                return

        if not hidden_layers and num_layers > 0:
             messagebox.showerror("Invalid Input", "Neurons per Layer cannot be empty if Number of Layers is > 0.")
             return
        
        if hidden_layers and num_layers == 0:
            messagebox.showerror("Input Mismatch", "Number of layers is 0, but 'Neurons per Layer' is not empty.")
            return

        try:
            learning_rate = float(self.learning_rate_input.get())
            if learning_rate <= 0:
                messagebox.showerror("Invalid Input", "Learning rate must be a positive number.")
                return
        except ValueError:
            messagebox.showerror("Invalid Input", "Learning rate must be a numeric value.")
            return

        try:
            epoch_count = int(self.epochs_input.get())
            if epoch_count <= 0:
                messagebox.showerror("Invalid Input", "Number of epochs must be a positive integer.")
                return
        except ValueError:
            messagebox.showerror("Invalid Input", "Number of epochs must be an integer value.")
            return
        
        chosen_activation = self.activation_func_choice.get()
        is_bias_enabled = self.use_bias_checkbox.get()

        try:
            processor = DataProcessor()
            X_train, X_test, y_train, y_test = processor.get_processed_data()

            input_size = X_train.shape[1]
            output_size = y_train.shape[1]
            layer_sizes = [input_size] + hidden_layers + [output_size]

            mlp = MLP(
                layer_sizes=layer_sizes,
                learning_rate=learning_rate,
                epochs=epoch_count,
                activation_func=chosen_activation,
                use_bias=is_bias_enabled
            )
            
            mlp.fit(X_train, y_train)

            accuracy, confusion_matrix_data = mlp.evaluate(X_test, y_test)
            accuracy_percent = accuracy * 100

            self.test_accuracy_label.config(text=f"Test Accuracy: {accuracy_percent:.2f}%")
            self.plot_confusion_matrix(confusion_matrix_data, processor.label_encoder_species.classes_)
            
            messagebox.showinfo("Success", f"Training complete.\nTest Accuracy: {accuracy_percent:.2f}%")

        except Exception as e:
            messagebox.showerror("Runtime Error", f"An error occurred during MLP execution: {e}")

    def plot_confusion_matrix(self, confusion_matrix_data, species_names):
        if self.plot_canvas:
            self.plot_canvas.get_tk_widget().destroy()

        figure, axis = plt.subplots(figsize=(5, 4))
        sns.heatmap(confusion_matrix_data, annot=True, fmt='d', cmap='Blues',
                    xticklabels=species_names,
                    yticklabels=species_names,
                    ax=axis)
        axis.set_xlabel('Predicted')
        axis.set_ylabel('Actual')
        axis.set_title('Confusion Matrix')
        
        self.plot_canvas = FigureCanvasTkAgg(figure, master=self.matrix_plot_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    main_menu = tk.Tk()
    app = GUI(main_menu)
    main_menu.mainloop()