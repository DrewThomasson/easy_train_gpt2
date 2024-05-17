# easy_train_gpt2
A easy way to train a custom GPT2 model with a singel prompt by utilizing ollama 



# llm_qna_database_generator
makes a llm with a specific prompt provide a new answer to all the input s to a input and response dataset in csv file format


-Run the `Ollama_dataset.py` file and itll use the base prompt charcter to reenter all the answer fields in the given llm input output dataset in csv.
-At the moment it is set by default for a batman prompt that makes all the swers all batmany


# Pip installs

`pip install PyQt5 pandas tqdm ollama
`

# Make sure you have Ollama installed on your computer also
https://ollama.com


# Pic of the gui and terminal when running

<img width="571" alt="image" src="https://github.com/DrewThomasson/llm_qna_database_generator/assets/126999465/cbf1e80a-71f8-4b18-964d-6b129ab76743">






example system prompt:

`You are batman you will alwsy talk in a dark gloomy tone, you will alwasy redirect the conversation to ebing batman, being an orphan and fighting your many enemies, be creative.  you will also throw in a last thing about how great the tyler perry movie is but its nothing in comparision to JUSTICE`




# To train gpt2 
-run `python train_gpt2`

# requirmed pip installs 
`pip install datasets transformers torch `
`pip install accelerate -U
`

# To test the now trained gpt2 model run 
`test_trained_gpt2.py`


# for the full run gui
`python full_run.py`
make sure you have thisese pips installed before taht and ollama installed with llama3 pulled already
`pip install -U PyQt5 pandas tqdm datasets transformers accelerate ollama
`
