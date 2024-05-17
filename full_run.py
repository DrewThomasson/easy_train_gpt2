import sys
import os
import pandas as pd
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget,
    QLineEdit, QProgressBar, QAction, QSlider, QLabel, QMessageBox, QTextEdit, QComboBox
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt, QMutex, QWaitCondition
from PyQt5.QtGui import QIcon
from tqdm import tqdm
import ollama
from datasets import load_dataset
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, DataCollatorForLanguageModeling, Trainer, TrainingArguments

class Worker(QThread):
    update_progress = pyqtSignal(int)
    finished = pyqtSignal()

    def __init__(self, df, system_prompt, num_rows, model_name):
        super().__init__()
        self.df = df
        self.system_prompt = system_prompt
        self.num_rows = num_rows
        self.model_name = model_name
        self.running = True
        self.paused = False
        self.mutex = QMutex()
        self.condition = QWaitCondition()

    def run(self):
        if 'prompt' not in self.df.columns:
            print("Error: DataFrame does not contain the 'prompt' column.")
            return

        for index, row in tqdm(self.df.iterrows(), total=self.num_rows):
            self.mutex.lock()
            while self.paused:
                self.condition.wait(self.mutex)
            self.mutex.unlock()

            if index >= self.num_rows:
                break
            if not self.running:
                break
            response = self.get_ollama_response(row['prompt'])
            self.df.at[index, 'output'] = response
            self.update_progress.emit(int((index + 1) / self.num_rows * 100))

        self.df.iloc[:self.num_rows].to_csv('filled_qna_dataset.csv', index=False)
        print("Dataset processing complete. Updated dataset saved as 'filled_qna_dataset.csv'.")
        self.finished.emit()

    def get_ollama_response(self, prompt):
        response = ollama.chat(model=self.model_name, messages=[
            {
                'role': 'user',
                'content': f'{self.system_prompt} {prompt}',
            },
        ])
        return response['message']['content']

    def stop(self):
        self.running = False
        self.paused = False
        self.condition.wakeAll()

    def pause(self):
        self.paused = True

    def resume(self):
        self.paused = False
        self.condition.wakeAll()

class TrainWorker(QThread):
    def run(self):
        dataset_path = 'filled_qna_dataset.csv'
        output_dir = 'output'
        data_files = {"train": dataset_path}
        dataset = load_dataset('csv', data_files=data_files)

        tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        def preprocess_function(examples):
            concatenated_examples = [p + tokenizer.eos_token + o for p, o in zip(examples['prompt'], examples['output'])]
            return tokenizer(concatenated_examples, truncation=True, max_length=512, padding="max_length", return_tensors="pt")

        tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=['prompt', 'output'])

        model = GPT2LMHeadModel.from_pretrained('gpt2')

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=10,
            per_device_train_batch_size=5,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir='./logs',
            save_strategy="no",
            save_total_limit=1,
            load_best_model_at_end=True,
            evaluation_strategy="no"
        )

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, return_tensors="pt")

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            data_collator=data_collator,
        )

        trainer.train()

        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        for root, dirs, files in os.walk(output_dir):
            for file in files:
                print(os.path.join(root, file))

class TestWorker(QThread):
    update_text = pyqtSignal(str)

    def __init__(self, model_path, prompt):
        super().__init__()
        self.prompt = prompt
        self.model_path = model_path

    def run(self):
        model, tokenizer = self.load_model(self.model_path)
        response = self.generate_response(model, tokenizer, self.prompt)
        self.update_text.emit(f'You: {self.prompt}\nModel: {response}\n')

    def load_model(self, model_path):
        model = GPT2LMHeadModel.from_pretrained(model_path)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
        return model, tokenizer

    def generate_response(self, model, tokenizer, prompt):
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
        self.model = None
        self.tokenizer = None
        self.dark_mode = False
        self.df = pd.read_csv('unfilled_qna_dataset.csv')
        self.init_ui()
        self.init_menu()
        self.set_stylesheet()

    def init_ui(self):
        self.setGeometry(200, 200, 600, 400)
        self.setWindowTitle('Data Processor GUI')

        layout = QVBoxLayout()

        self.model_select = QComboBox(self)
        self.model_select.addItems([
            "llama3", "phi3", "wizardlm2", "mistral", "gemma", "mixtral", "llama2",
            "codegemma", "command-r", "command-r-plus", "llava", "dbrx", "codellama",
            "qwen", "dolphin-mixtral", "llama2-uncensored", "deepseek-coder",
            "mistral-openorca", "nomic-embed-text", "dolphin-mistral", "phi",
            "orca-mini", "nous-hermes2", "zephyr", "llama2-chinese",
            "wizard-vicuna-uncensored", "starcoder2", "vicuna", "tinyllama",
            "openhermes", "openchat", "starcoder", "dolphin-llama3", "yi",
            "tinydolphin", "wizardcoder", "stable-code", "mxbai-embed-large",
            "neural-chat", "phind-codellama", "wizard-math", "starling-lm",
            "falcon", "dolphincoder", "orca2", "nous-hermes", "stablelm2",
            "sqlcoder", "dolphin-phi", "solar", "deepseek-llm", "yarn-llama2",
            "codeqwen", "bakllava", "samantha-mistral", "all-minilm",
            "medllama2", "llama3-gradient", "wizardlm-uncensored", "nous-hermes2-mixtral",
            "xwinlm", "stable-beluga", "codeup", "wizardlm", "yarn-mistral",
            "everythinglm", "meditron", "llama-pro", "magicoder", "stablelm-zephyr",
            "nexusraven", "codebooga", "mistrallite", "wizard-vicuna", "llama3-chatqa",
            "snowflake-arctic-embed", "goliath", "open-orca-platypus2", "llava-llama3",
            "moondream", "notux", "megadolphin", "duckdb-nsql", "notus", "alfred",
            "llava-phi3", "falcon2"
        ])
        layout.addWidget(self.model_select)

        self.prompt_input = QLineEdit(self)
        self.prompt_input.setPlaceholderText('Enter your system prompt here...')
        layout.addWidget(self.prompt_input)

        self.slider_label = QLabel(f"Number of rows to fill: 0 / {len(self.df)}", self)
        layout.addWidget(self.slider_label)

        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(self.df))
        self.slider.valueChanged.connect(self.update_slider_label)
        layout.addWidget(self.slider)

        self.generate_button = QPushButton('Generate Dataset')
        self.generate_button.clicked.connect(self.start_processing)
        layout.addWidget(self.generate_button)

        self.train_button = QPushButton('Train Model')
        self.train_button.clicked.connect(self.train_model)
        layout.addWidget(self.train_button)

        self.test_button = QPushButton('Test Model')
        self.test_button.clicked.connect(self.setup_test_model)
        layout.addWidget(self.test_button)

        self.pause_button = QPushButton('Pause')
        self.pause_button.clicked.connect(self.pause_processing)
        self.pause_button.setVisible(False)
        layout.addWidget(self.pause_button)

        self.stop_button = QPushButton('Stop Now')
        self.stop_button.clicked.connect(self.stop_processing)
        layout.addWidget(self.stop_button)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_bar)

        self.chat_display = QTextEdit(self)
        self.chat_display.setReadOnly(True)
        self.chat_display.setVisible(False)
        layout.addWidget(self.chat_display)

        self.chat_input = QLineEdit(self)
        self.chat_input.setPlaceholderText('Enter your prompt here...')
        self.chat_input.setVisible(False)
        self.chat_input.returnPressed.connect(self.send_chat)
        layout.addWidget(self.chat_input)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def init_menu(self):
        toggle_theme_action = QAction(QIcon(), 'Toggle Theme', self)
        toggle_theme_action.triggered.connect(self.toggle_theme)
        self.toolbar = self.addToolBar('Toggle Theme')
        self.toolbar.addAction(toggle_theme_action)

    def set_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #f5f5f5;
                color: #333;
                font-family: Arial, sans-serif;
                font-size: 14px;
            }
            QPushButton {
                background-color: #0078d7;
                color: #fff;
                border: none;
                padding: 10px;
                margin: 5px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #005fa3;
            }
            QLineEdit, QTextEdit, QProgressBar {
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 5px;
            }
            QLabel {
                margin: 5px;
            }
            QSlider::groove:horizontal {
                border: 1px solid #bbb;
                background: #f5f5f5;
                height: 10px;
                border-radius: 4px;
            }
            QSlider::sub-page:horizontal {
                background: #0078d7;
                border: 1px solid #777;
                height: 10px;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #fff;
                border: 1px solid #0078d7;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
        """)

    def toggle_theme(self):
        if self.dark_mode:
            self.set_stylesheet()
            self.dark_mode = False
        else:
            self.setStyleSheet("""
                QMainWindow, QWidget {
                    background-color: #333;
                    color: #eee;
                    font-family: Arial, sans-serif;
                    font-size: 14px;
                }
                QPushButton {
                    background-color: #444;
                    color: #fff;
                    border: none;
                    padding: 10px;
                    margin: 5px;
                    border-radius: 5px;
                }
                QPushButton:hover {
                    background-color: #555;
                }
                QLineEdit, QTextEdit, QProgressBar {
                    border: 1px solid #555;
                    border-radius: 5px;
                    padding: 5px;
                }
                QLabel {
                    margin: 5px;
                }
                QSlider::groove:horizontal {
                    border: 1px solid #bbb;
                    background: #333;
                    height: 10px;
                    border-radius: 4px;
                }
                QSlider::sub-page:horizontal {
                    background: #444;
                    border: 1px solid #777;
                    height: 10px;
                    border-radius: 4px;
                }
                QSlider::handle:horizontal {
                    background: #fff;
                    border: 1px solid #444;
                    width: 18px;
                    margin: -2px 0;
                    border-radius: 9px;
                }
            """)
            self.dark_mode = True

    def start_processing(self):
        system_prompt = self.prompt_input.text()
        num_rows = self.slider.value()
        model_name = self.model_select.currentText()
        if not system_prompt or num_rows == 0:
            self.show_alert("Please provide a system prompt and select the number of rows to fill.")
            return
        self.worker = Worker(self.df, system_prompt, num_rows, model_name)
        self.worker.update_progress.connect(self.update_progress_bar)
        self.worker.finished.connect(self.on_generation_finished)
        self.worker.start()
        self.generate_button.setVisible(False)
        self.pause_button.setVisible(True)
        self.slider.setEnabled(False)

    def update_progress_bar(self, value):
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(f"{value}%")

    def pause_processing(self):
        if self.worker:
            if self.worker.paused:
                self.worker.resume()
                self.pause_button.setText('Pause')
            else:
                self.worker.pause()
                self.pause_button.setText('Resume')

    def stop_processing(self):
        if self.worker:
            self.worker.stop()
            self.worker.wait()
            self.on_generation_finished()

    def on_generation_finished(self):
        self.pause_button.setVisible(False)
        self.generate_button.setVisible(True)
        self.slider.setEnabled(True)
        self.show_alert("Dataset generation complete. The updated dataset has been saved as 'filled_qna_dataset.csv'.")

    def update_slider_label(self, value):
        self.slider_label.setText(f"Number of rows to fill: {value} / {len(self.df)}")

    def train_model(self):
        self.train_worker = TrainWorker()
        self.train_worker.start()
        self.train_worker.finished.connect(self.on_training_complete)

    def on_training_complete(self):
        QMessageBox.information(self, 'Training Complete', 'The model has been trained successfully.')
        self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self):
        self.model = GPT2LMHeadModel.from_pretrained('output')
        self.tokenizer = GPT2TokenizerFast.from_pretrained('output')

    def setup_test_model(self):
        self.chat_display.setVisible(True)
        self.chat_input.setVisible(True)
        self.chat_input.setFocus()

    def send_chat(self):
        prompt = self.chat_input.text()
        if prompt:
            self.chat_input.clear()
            self.test_worker = TestWorker('output', prompt)
            self.test_worker.update_text.connect(self.update_chat_display)
            self.test_worker.start()

    def update_chat_display(self, text):
        self.chat_display.append(text)

    def show_alert(self, message):
        alert = QMessageBox()
        alert.setText(message)
        alert.exec_()

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
    ex.show()
    sys.exit(app.exec_())
