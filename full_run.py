import sys
import os
import pandas as pd
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QLineEdit, QProgressBar, QAction, QSlider, QLabel, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QIcon
from tqdm import tqdm
import ollama
from datasets import load_dataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments

class Worker(QThread):
    update_progress = pyqtSignal(int)

    def __init__(self, df, system_prompt, num_rows):
        super().__init__()
        self.df = df
        self.system_prompt = system_prompt
        self.num_rows = num_rows
        self.running = True

    def run(self):
        if 'prompt' not in self.df.columns:
            print("Error: DataFrame does not contain the 'prompt' column.")
            return  # Exit the thread if the required column is missing

        for index, row in tqdm(self.df.iterrows(), total=self.num_rows):
            if index >= self.num_rows:
                break
            if not self.running:
                break
            response = self.get_ollama_response(row['prompt'])
            self.df.at[index, 'output'] = response
            self.update_progress.emit(int((index + 1) / self.num_rows * 100))
        
        self.df.iloc[:self.num_rows].to_csv('filled_qna_dataset.csv', index=False)
        print("Dataset processing complete. Updated dataset saved as 'filled_qna_dataset.csv'.")

    def get_ollama_response(self, prompt):
        response = ollama.chat(model='llama3', messages=[
            {
                'role': 'user',
                'content': f'{self.system_prompt} {prompt}',
            },
        ])
        return response['message']['content']

    def stop(self):
        self.running = False

class TrainWorker(QThread):
    update_progress = pyqtSignal(int)

    def __init__(self):
        super().__init__()

    def run(self):
        # Parameters
        dataset_path = 'filled_qna_dataset.csv'
        output_dir = 'output'

        # Load the dataset
        data_files = {"train": dataset_path}
        dataset = load_dataset('csv', data_files=data_files)

        # Initialize the tokenizer and set the pad token
        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        def preprocess_function(examples):
            # Concatenate prompt and output into a single text entry for each example
            concatenated_examples = [p + tokenizer.eos_token + o for p, o in zip(examples['prompt'], examples['output'])]
            return tokenizer(concatenated_examples, truncation=True, max_length=512, padding="max_length", return_tensors="pt")

        tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=['prompt', 'output'])

        # Load the pre-trained GPT-2 model
        model = GPT2LMHeadModel.from_pretrained('gpt2')

        # Define training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=10,
            per_device_train_batch_size=5,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            save_strategy="no",  # No model saving during training
            save_total_limit=1,  # Maximum number of models to keep
            load_best_model_at_end=True,  # Optional: Load the best model at the end of training
            evaluation_strategy="no"  # Assuming no evaluation is performed
        )

        # Initialize the Data Collator for causal language modeling
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")

        # Initialize the Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            data_collator=data_collator,
        )

        # Start training
        trainer.train()

        # Save model and tokenizer
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        # Verify all files are saved
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                print(os.path.join(root, file))

class TestWorker(QThread):
    def __init__(self, prompt):
        super().__init__()
        self.prompt = prompt
        self.model_path = "output"  # Replace with your model directory path

    def run(self):
        model, tokenizer = self.load_model(self.model_path)
        response = self.generate_response(model, tokenizer, self.prompt)
        print("Model:", response)

    def load_model(self, model_path):
        # Load the fine-tuned model and tokenizer
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
        return model, tokenizer

    def generate_response(self, model, tokenizer, prompt):
        # Encode the prompt and generate response
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        outputs = model.generate(inputs, max_length=200, num_return_sequences=1)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response

class AppWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.train_worker = None
        self.test_worker = None
        self.dark_mode = False
        self.init_ui()
        self.init_menu()

    def init_ui(self):
        self.setGeometry(200, 200, 400, 400)
        self.setWindowTitle('Data Processor GUI')
        
        layout = QVBoxLayout()

        self.prompt_input = QLineEdit(self)
        self.prompt_input.setPlaceholderText('Enter your system prompt here...')
        layout.addWidget(self.prompt_input)

        self.slider_label = QLabel('Number of rows: 0', self)
        layout.addWidget(self.slider_label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(0)  # Will be set dynamically based on the number of rows in the CSV
        self.slider.valueChanged.connect(self.update_slider_label)
        layout.addWidget(self.slider)

        self.generate_button = QPushButton('Generate Dataset')
        self.generate_button.clicked.connect(self.start_processing)
        layout.addWidget(self.generate_button)

        self.train_button = QPushButton('Train Model')
        self.train_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_button)

        self.test_button = QPushButton('Test Model')
        self.test_button.clicked.connect(self.test_model)
        layout.addWidget(self.test_button)

        self.pause_button = QPushButton('Pause')
        self.pause_button.clicked.connect(self.pause_processing)
        self.pause_button.setVisible(False)  # Hide the pause button initially
        layout.addWidget(self.pause_button)

        self.stop_button = QPushButton('Stop Now')
        self.stop_button.clicked.connect(self.stop_processing)
        layout.addWidget(self.stop_button)

        self.progress_bar = QProgressBar(self)
        layout.addWidget(self.progress_bar)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def init_menu(self):
        toggle_theme_action = QAction(QIcon(), 'Toggle Theme', self)
        toggle_theme_action.triggered.connect(self.toggle_theme)
        self.toolbar = self.addToolBar('Toggle Theme')
        self.toolbar.addAction(toggle_theme_action)

    def toggle_theme(self):
        if self.dark_mode:
            self.setStyleSheet("")
            self.dark_mode = False
        else:
            self.setStyleSheet("""
                QMainWindow, QWidget, QPushButton, QLineEdit, QProgressBar, QToolBar {
                    background-color: #333;
                    color: #eee;
                    border: 1px solid #555;
                }
                QLineEdit {
                    border: 2px solid #555;
                }
                QProgressBar::chunk {
                    background-color: #44a;
                }
            """)
            self.dark_mode = True

    def start_processing(self):
        system_prompt = self.prompt_input.text()
        self.df = pd.read_csv('unfilled_qna_dataset.csv')
        num_rows = self.slider.value()
        self.worker = Worker(self.df, system_prompt, num_rows)
        self.worker.update_progress.connect(self.progress_bar.setValue)
        self.worker.start()
        self.pause_button.setVisible(True)
        self.slider.setEnabled(False)  # Disable the slider
        self.generate_button.setEnabled(False)  # Disable the generate button

    def pause_processing(self):
        if self.worker:
            self.worker.running = not self.worker.running
            self.pause_button.setText('Resume' if not self.worker.running else 'Pause')

    def stop_processing(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.pause_button.setVisible(False)
            self.slider.setEnabled(True)  # Re-enable the slider
            self.generate_button.setEnabled(True)  # Re-enable the generate button

    def update_slider_label(self, value):
        self.slider_label.setText(f'Number of rows: {value}')

    def train_model(self):
        self.train_worker = TrainWorker()
        self.train_worker.start()
        self.train_worker.finished.connect(self.on_training_complete)

    def on_training_complete(self):
        QMessageBox.information(self, 'Training Complete', 'The model has been trained successfully.')

    def test_model(self):
        prompt, ok = QInputDialog.getText(self, 'Test Model', 'Enter your prompt:')
        if ok:
            self.test_worker = TestWorker(prompt)
            self.test_worker.start()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait()
        if self.train_worker and self.train_worker.isRunning():
            self.train_worker.terminate()
        if self.test_worker and self.test_worker.isRunning():
            self.test_worker.terminate()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = AppWindow()
    ex.df = pd.read_csv('unfilled_qna_dataset.csv')
    ex.slider.setMaximum(len(ex.df))
    ex.show()
    sys.exit(app.exec_())
