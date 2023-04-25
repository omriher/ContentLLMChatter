# LLaMaContentChatter
This code is a Python script for generating responses to questions based on a set of PDF files. It uses the Llama language model from the HuggingFace Transformers library.

The code has a custom implementation of an appropriate `Embeddings` class.

### Prerequisites

To run this script, you will need to have the following libraries installed:

- transformers
- torch
- bitsandbytes
- accelerate
- sentencepiece
- langchain

### Usage

To run the script, use the following command:

```
python LLaMaContentChatter.py -m MODEL_NAME -f FOLDER -q QUESTION
```
Where `MODEL_NAME` is the name of the HuggingFace model to use, `FOLDER` is the path to the folder containing the PDF files to analyze, and `QUESTION` is the question to ask the language model - use quotation marks `""`.

For Example:
```
python LLaMaContentChatter.py -m samwit/koala-7b -f ./docs -q "how does the pseudocode look like? show an example"                      
```


### Functionality

The script does the following:

1. Loads the specified LLaMa model and tokenizer.
2. Generates a language model pipeline using the model and tokenizer.
3. Creates a vector store from the files in the specified folder using a custom implementation of an embeddings class.
4. Queries the vector store with the specified question using the language model pipeline and returns the response.

### Credits

This script was created by Omri Herscovici. It uses the Llama language model from the HuggingFace Transformers library.
