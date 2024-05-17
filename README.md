# 🚀 Easy Train GPT-2
A simple way to train a custom GPT-2 model with a single prompt using Ollama.

## 📦 Required Pip Installs
```sh
pip install -U PyQt5 pandas tqdm datasets transformers torch accelerate ollama
```

## ▶️ Run the Full GUI
```sh
python full_run.py
```

![image](https://github.com/DrewThomasson/easy_train_gpt2/assets/126999465/336b51a0-10f5-4a7b-a15b-d6bc06b7bf38)

<details>
  <summary>📝 Extra Info About Other Files</summary>

  # llm_qna_database_generator
  Creates a LLM with a specific prompt to provide new answers to all inputs in a dataset in CSV file format.

  - Run the `Ollama_dataset.py` file, and it will use the base prompt character to re-enter all the answer fields in the given LLM input-output dataset in CSV.

  ## 📦 Pip Installs
  ```sh
  pip install PyQt5 pandas tqdm ollama
  ```

  ## 🛠️ Make Sure You Have Ollama Installed
  [Ollama Installation](https://ollama.com)

  ## 📸 GUI and Terminal When Running
  ![image](https://github.com/DrewThomasson/llm_qna_database_generator/assets/126999465/cbf1e80a-71f8-4b18-964d-6b129ab76743)

  ### 💬 Example System Prompt:
  ```markdown
  You are Batman. You will always talk in a dark, gloomy tone. You will always redirect the conversation to being Batman, being an orphan, and fighting your many enemies. Be creative. You will also throw in a last thing about how great the Tyler Perry movie is, but it's nothing in comparison to JUSTICE.
  ```

  ## 🦾 To Train GPT-2
  Run:
  ```sh
  python train_gpt2
  ```

  ## 📦 Required Pip Installs
  ```sh
  pip install -U datasets transformers torch PyQt5 pandas tqdm ollama
  pip install accelerate -U
  ```

  ## 🧪 To Test the Now Trained GPT-2 Model
  Run:
  ```sh
  python test_trained_gpt2.py
  ```

</details>
