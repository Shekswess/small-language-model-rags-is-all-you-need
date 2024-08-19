# small-language-model-rags-is-all-you-need

This is a repository that contains the code for the experiments conducted for the project called "Small Language Model RAGs is all you need". The project aims to showcase the capabilities of building RAG systems on top of smaller language models like Gemma2 9B, Llama 3 8B, Mistral 7B, and others. The idea of the project is how we can leverage the capabilities of smaller language models with some smart prompt engineering and some interesting inovative ideas to have results that are comparable or in some cases better than RAG systems built on top of larger language models like GPT-4o, Claude 3.5 Sonnet, and others.

For the experiments we used two different approaches or with other words types of RAG systems:
- Simple RAG - The classic RAG system pipeline
- Mixture of RAG (Mixture RAG) - RAG system pipeline inspired by the implementation of Mixture of Agents (MoA)

## Libraries used in this project:
- [Langfuse](https://www.langfuse.com/)
- [Langchain](https://www.langchain.com/)
- [Ragas](https://docs.ragas.io/en/stable/)
- [Pydantic](https://pydantic-docs.helpmanual.io/)

## LLM providers:
- [AWS Bedrock](https://aws.amazon.com/bedrock/)
- [Groq](https://www.groq.com/)
- [OpenAI](https://www.openai.com/)

## Requirements:
- [Python](https://www.python.org/downloads/)
- [pip](https://pip.pypa.io/en/stable/installation/)
- [Docker & Docker Compose](https://docs.docker.com/get-docker/)
- [AWS Account](https://aws.amazon.com/)
- [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html)
- [AWS SSO account login](https://docs.aws.amazon.com/singlesignon/latest/userguide/getting-started.html)
- [OpenAI API key](https://platform.openai.com/docs/guides/authentication)
- [Groq API key](https://www.groq.com/)

## Environment Variables:
Create a `.env` file in the root directory of the project and add the following environment variables:
```bash
BEDROCK_REGION_NAME = 'BEDROCK_REGION_NAME'
BEDROCK_CREDENTIALS_PROFILE_NAME = 'BEDROCK_CREDENTIALS_PROFILE_NAME'
OPENAI_API_KEY = 'OPENAI_API_KEY'
GROQ_API_KEY = 'GROQ_API_KEY'
LANGFUSE_SECRET_KEY = 'SECRET_KEY'
LANGFUSE_PUBLIC_KEY = 'PUBLIC_KEY'
LANGFUSE_HOST = 'HOST'
LANGFUSE_SCORE_URL = 'HOST_SCORE_URL'
```
The environment variables are used to authenticate the AWS Bedrock, OpenAI, Groq, and Langfuse APIs.

## Configuration for experiments

We have two different types of RAG system pipelines which were used for the experiments:
- Simple RAG - The classic RAG system pipeline
- Mixture of RAG (Mixture RAG) - RAG system pipeline inspired by the implementation of Mixture of Agents (MoA)

The example configuration for the Simple RAG looks like:
```yaml
experiment_name: "simple_rag"

llm:
  provider: "bedrock" 
  model_spec:
    model_id: "anthropic.claude-3-haiku-20240307-v1:0"
    model_kwargs: 
      max_tokens: 4096
      temperature: 0.1
      top_k: 250
      top_p: 1
      stop_sequences: ["\n\nHuman"]
  prompt:
    system_message: "Act like a Machine Learning Teacher"
    user_message: "Ask a question or provide a prompt"

embedder:
  model_id: "amazon.titan-embed-text-v2:0"
  model_kwargs: 
    dimensions: 512
    normalize: True

retriever:
  search_type: "similarity"
  retriever_kwargs:
    k: 5

chunker:
  chunk_size: 1500
  chunk_overlap: 100

data:
  path: "./data/raw/"

vector_store:
  path: "./data/database_1500_100"
```

The example configuration for the Mixture RAG looks like:
```yaml
experiment_name: "mixture_rag"

layers:
  - layer_type: "rag"
    layer_spec:
      - llm:
          provider: "bedrock"
          model_spec:
            model_id: "anthropic.claude-3-haiku-20240307-v1:0"
            model_kwargs:
              max_tokens: 4096
              temperature: 0.1
              top_k: 500
              top_p: 1
              stop_sequences: ["\n\nHuman"]
          prompt:
            system_message: "Act like a Machine Learning Expert"
            user_message: "Ask a question or provide a prompt"
      - llm:
          provider: "groq"
          model_spec:
            model_name: "mixtral-8x7b-32768"
            temperature: 0.1
            max_tokens: 4096
          prompt:
            system_message: "Act like a Machine Learning Beginner"
            user_message: "Ask a question or provide a prompt"
      - llm:
          provider: "groq"
          model_spec:
            model_name: "mixtral-8x7b-32768"
            temperature: 0.1
            max_tokens: 4096
          prompt:
            system_message: "Act like a Machine Learning Teacher"
            user_message: "Ask a question or provide a prompt"
  - layer_type: "aggregator"
    layer_spec:
      - llm:
          provider: "openai"
          model_spec:
            model: "gpt-4o"
            temperature: 0.1
            max_tokens: 4096
          prompt:
            system_message: |
              You have been provided with a set of responses from various open-source models to the latest user query. 
              Your task is to synthesize these responses into a single, high-quality response. 
              It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. 
              Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. 
              Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.
            user_message: "Ask a question or provide a prompt"

embedder:
  model_id: "amazon.titan-embed-text-v2:0"
  model_kwargs:
    dimensions: 512
    normalize: true

retriever:
  search_type: "similarity"
  retriever_kwargs:
    k: 2

chunker:
  chunk_size: 1500
  chunk_overlap: 100

data:
  path: "./data"

vector_store:
  path: "./data/database_1500_100"
```

The validation of the configuration files is done with Pydantic(more info about it can be seen in the configuration models `src/configuration/configuration_model.py`)

The configuration files of a Simple RAG pipeline must have the following structure:
```
- experiment_name: str
- llm: dict
    - provider: str
    - model_spec: dict
    - prompt: dict
      - system_message: str
      - user_message: str
- embedder: dict
    - model_id: str
    - model_kwargs: dict
- retriever: dict
    - search_type: str
    - retriever_kwargs: dict
- chunker: dict
    - chunk_size: int
    - chunk_overlap: int
- data: dict
    path: str
- vector_store: dict
    path: str (for additional validation the path name must include the chunk_size and chunk_overlap)
```

The configuration files of a Mixture RAG pipeline must have the following structure:
```
- experiment_name: str
- layers: list (it has to have two layers one the rag llms and the other the aggregator llm)
    - layer_type: str
    - layer_spec: list
        - llm: dict
            - provider: str
            - model_spec: dict
            - prompt: dict
              - system_message: str
              - user_message: str
- embedder: dict
    - model_id: str
    - model_kwargs: dict
- retriever: dict
    - search_type: str
    - retriever_kwargs: dict
- chunker: dict
    - chunk_size: int
    - chunk_overlap: int
- data: dict
    path: str
- vector_store: dict
    path: str (for additional validation the path name must include the chunk_size and chunk_overlap)
```

The model_spec is different for each provider:
```
- Bedrock
    - model_id: str
    - model_kwargs: dict
- Groq
    - model_name: str
    - temperature: float
    - max_tokens: int
- OpenAI
    - model: str
    - temperature: float
    - max_tokens: int
```


## How to run the experiments:

1. Clone the repository
2. Create a `.env` file in the root directory of the project and add the environment variables(see the Environment Variables section)
3. Create a configuration file for the experiment you want to run(see the Configuration for experiments section) and put in the `src/configuration` folder
4. Setup the Langfuse Server:
    - 4a. Use their Cloud Service hosted Langfuse
    - 4b. Use the Docker Compose file to run the Langfuse Server locally
5. Modify the `execute_pipeline.py` file with the configuration file you want to run and the prompt templates you want to use.
6. Run the `execute_pipeline.py` file with the following command:
```bash
python execute_pipeline.py
```
7. Generate the results by running the `extract_results.py` file with the following command:
```bash
python extract_results.py
```
8. The results will be saved in the `results` folder and you can analyze them.


## Experiments

In total there were 28 experiments both with Simple RAG and Mixture RAG pipelines.

### Dataset

The dataset used for the experiments is a collection of research papers in the field of Natural Language Processing (NLP), to be exact in the field of Large Language Models (LLMs). The dataset consists of 14 most cited papers in the field of NLP and LLMs. Questions from these papers were used as the evaluation dataset for the experiments.

### Experimental Pipeline Setup

All the experimental pipelines share these common components:
- **Chunker**: The dataset is chunked into smaller parts to be used for the experiments. The chunk size is 1500 and the chunk overlap is 100.
- **Embedder**: The Amazon Titan Embed Text model is used to embed the chunks of the dataset, with 512 vector dimensions.
- **Vector Store**: The embedded vectors are stored in a FAISS vector database for faster retrieval.
- **Retriever**: The retrieval of the most similar chunks is done using the FAISS vector database. The number of similar chunks retrieved is 5 and the search type is similarity.

The experimental pipelines differ in the LLMs used and the way of the LLMs are used/combined.

The LLM used in the pipelines are:
- gemma2-9b-it
- gemma-7b-it
- mistral-7b-instruct
- mixtral-8x7b-instruct
- llama-3-8b-instruct
- llama-3.1-8b-instruct
- llama-3-70b-instruct
- llama-3.1-70b-instruct
- llama-3.1-405b-instruct
- claude-3-haiku
- claude-3-sonnet
- claude-3-opus
- claude-3-5-sonnet
- gpt-4o
- gpt-4o-mini
- gpt-4-turbo

Each of the LLMs have specific instruction prompt templates that are used for the experiments. Those templates can be found on:
- [Prompts Engineering Guide](https://www.promptingguide.ai/)
- [Ollama](https://ollama.ai/)
- [Anthropic Prompt Engineering](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)

All of the prompts used in the experiments are stored in the `src/constants/prompts.py` file.

#### Simple RAG Pipeline

The Simple RAG pipeline uses a single LLM to generate the responses. This is how the Simple RAG looks:

<!Image>

For all the experiments the system and the user messages are the same:

```
system_message:
    "You are an assistant that has a lot of knowledge about Large Language Models.
    Answer the user's question in a way that is easy to understand and informative.
    Use the provided context to generate a response that is relevant and accurate.
    You are an assistant that has a lot of knowledge about Large Language Models.
    Answer the user's question in a way that is easy to understand and informative.
    Use the provided context to generate a response that is relevant and accurate."

user_message: "Please answer my question based on the provided context:"
```

#### Mixture RAG Pipeline

The Mixture RAG pipeline mostly is like the Simple RAG pipeline but in the Generator we basically trigger multiple LLMs(Simple RAGs with the same prompt system and user messages previsoly defined) to generate the responses and those response are the aggregated by another LLM. This is how the Mixture RAG looks:

<!Image>

There are three different system and user messages combinations used for the experiments, for the aggregation LLM:

- One combination is really similar to the one used in the Mixture of Agents (MoA) implementation:
```
system_message: 
    You have been provided with a set of responses from various small language models to the latest user query. 
    Your task is to synthesize these responses into a single, high-quality response. 
    It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. 
    Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply. 
    Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.
    
user_message: "Please synthesize the responses from the small language models and give me only the most accurate information."
```

- Second combination is a bit modified version of the first one:
```
system_message: 
    You have been provided with a set of responses from various small language models to the latest user query. 
    The responses of the small language models are based on the context provided in the user query.
    Your task is to create a single, high-quality response based on the responses of the small language models.
    You should perform something like a ensemble model based on majority voting.
    Your response should be very accurate and informative, while keeping the faithfulness and relevance to the previous responses.

user_message: "Please generate a single response based on the provided responses:"
```

- Third combination is basically making the aggregator LLM to choose the best response from the generated responses(thought):
```
system_message: 
    You have been provided with a set of responses from various small language models to the latest user query. 
    The responses of the small language models are based on the context provided in the user query.
    Your task is to choose the best response from the provided responses.
    You should choose the response by analyzing all available responses and selecting the one you think is the most accurate and informative.
    Keep in mind the response must be a high-quality response, while getting the most faithful and relevant information from the provided responses.
    When you have made your choice, make that your final response and do not provide any additional responses, like explanations or clarifications why you chose that response.

user_message: "Please choose a single response based on the provided responses:"
```

All the configurations for the experiments can be found in the `src/config` folder.

### Results and Conclusion

<h2>Question Mean Scores</h2>
<table>
  <thead>
    <tr>
      <th>Index</th>
      <th>Question</th>
      <th>Mean Score</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>1</td><td>How many stages are there in the development of the Llama 3 model?</td><td>0.939045</td></tr>
    <tr><td>2</td><td>Does Claude 3 models have vision capabilities?</td><td>0.925869</td></tr>
    <tr><td>3</td><td>Can the GPT-4 model accept both text and image inputs?</td><td>0.884999</td></tr>
    <tr><td>4</td><td>On what architecture the Gemma model is based on?</td><td>0.86232</td></tr>
    <tr><td>5</td><td>What is the difference between the Llama 2 and Llama 2-Chat ?</td><td>0.857979</td></tr>
    <tr><td>6</td><td>Is Mixtral based on the idea of a mixture of experts?</td><td>0.855282</td></tr>
    <tr><td>7</td><td>How many stages of training are in the GPT model?</td><td>0.848322</td></tr>
    <tr><td>8</td><td>What tokenizer is used in the Gemma2 model?</td><td>0.785732</td></tr>
    <tr><td>9</td><td>What is Mixture of Agents?</td><td>0.770606</td></tr>
    <tr><td>10</td><td>What are the two tasks in BERT?</td><td>0.768416</td></tr>
    <tr><td>11</td><td>How can attention be described in the Transformer?</td><td>0.743815</td></tr>
    <tr><td>12</td><td>What is sliding window attention?</td><td>0.741466</td></tr>
    <tr><td>13</td><td>What is optimizer is used for LLaMA?</td><td>0.643345</td></tr>
    <tr><td>14</td><td>On what architecture the GPT-3 model is based on?</td><td>0.583516</td></tr>
  </tbody>
</table>

<h2>Experiment Faithfulness</h2>
<table>
  <thead>
    <tr>
      <th>Index</th>
      <th>Experiment</th>
      <th>Faithfulness</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>1</td><td>simple-rag-llama-3.1-70b-instruct</td><td>0.961231</td></tr>
    <tr><td>2</td><td>simple-rag-llama-3.1-8b</td><td>0.957778</td></tr>
    <tr><td>3</td><td>simple-rag-llama-3.1-405b-instruct</td><td>0.945641</td></tr>
    <tr><td>4</td><td>mixture-rag-gemma2-9b-it-thought</td><td>0.924542</td></tr>
    <tr><td>5</td><td>simple-rag-gemma-7b-it</td><td>0.923677</td></tr>
    <tr><td>6</td><td>simple-rag-llama-3-8b</td><td>0.913214</td></tr>
    <tr><td>7</td><td>simple-rag-llama-3-70b</td><td>0.901136</td></tr>
    <tr><td>8</td><td>simple-rag-mixtral-8x7b-instruct</td><td>0.896447</td></tr>
    <tr><td>9</td><td>simple-rag-gpt-4o</td><td>0.895355</td></tr>
    <tr><td>10</td><td>mixture-rag-mixtral-8x7-instruct-thought</td><td>0.892727</td></tr>
    <tr><td>11</td><td>mixture-rag-mixtral-8x7-instruct-modified</td><td>0.882197</td></tr>
    <tr><td>12</td><td>simple-rag-mistral-7b-instruct</td><td>0.878027</td></tr>
  </tbody>
</table>

<h2>Experiment Answer Relevancy</h2>
<table>
  <thead>
    <tr>
      <th>Index</th>
      <th>Experiment</th>
      <th>Answer Relevancy</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>1</td><td>simple-rag-gpt-4o-mini</td><td>0.918347</td></tr>
    <tr><td>2</td><td>simple-rag-mistral-7b-instruct</td><td>0.914597</td></tr>
    <tr><td>3</td><td>mixture-rag-gemma2-9b-it-thought</td><td>0.910476</td></tr>
    <tr><td>4</td><td>simple-rag-claude-3.5-sonnet</td><td>0.90533</td></tr>
    <tr><td>5</td><td>simple-rag-gemma2-9b-it</td><td>0.905305</td></tr>
    <tr><td>6</td><td>mixture-rag-llama3.1-8b-instruct-thought</td><td>0.897726</td></tr>
    <tr><td>7</td><td>simple-rag-claude-3-opus</td><td>0.891054</td></tr>
    <tr><td>8</td><td>simple-rag-llama-3-70b</td><td>0.885328</td></tr>
    <tr><td>9</td><td>simple-rag-mixtral-8x7b-instruct</td><td>0.884369</td></tr>
    <tr><td>10</td><td>simple-rag-gpt-4o</td><td>0.884128</td></tr>
    <tr><td>11</td><td>simple-rag-claude-3-sonnet</td><td>0.874334</td></tr>
    <tr><td>12</td><td>mixture-rag-llama3.1-8b-instruct-modified</td><td>0.871686</td></tr>
    <tr><td>13</td><td>mixture-rag-gemma2-9b-it-modified</td><td>0.867729</td></tr>
  </tbody>
</table>

<h2>Experiment Context Utilization</h2>
<table>
  <thead>
    <tr>
      <th>Index</th>
      <th>Experiment</th>
      <th>Context Utilization</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>1</td><td>mixture-rag-llama3.1-8b-instruct</td><td>0.916667</td></tr>
    <tr><td>2</td><td>mixture-rag-mixtral-8x7-instruct-modified</td><td>0.916667</td></tr>
    <tr><td>3</td><td>mixture-rag-mixtral-8x7-instruct</td><td>0.913889</td></tr>
    <tr><td>4</td><td>simple-rag-mixtral-8x7b-instruct</td><td>0.908333</td></tr>
    <tr><td>5</td><td>simple-rag-mistral-7b-instruct</td><td>0.908333</td></tr>
    <tr><td>6</td><td>simple-rag-gpt-4o-mini</td><td>0.9</td></tr>
    <tr><td>7</td><td>mixture-rag-llama3.1-8b-instruct-modified</td><td>0.897222</td></tr>
    <tr><td>8</td><td>simple-rag-llama-3.1-405b-instruct</td><td>0.897222</td></tr>
    <tr><td>9</td><td>mixture-rag-gemma2-9b-it-thought</td><td>0.894444</td></tr>
    <tr><td>10</td><td>simple-rag-gemma-7b-it</td><td>0.888889</td></tr>
    <tr><td>11</td><td>simple-rag-gemma2-9b-it</td><td>0.883333</td></tr>
    <tr><td>12</td><td>simple-rag-claude-3-opus</td><td>0.883333</td></tr>
    <tr><td>13</td><td>mixture-rag-gemma2-9b-it-modified</td><td>0.869444</td></tr>
  </tbody>
</table>


## Project Structure:
```
.
├── config                                      # Configuration files
├── data                                        # Data files & Vector Database files
|   ├── database_1500_100                       # FAISS Vector Database files for database with 1500 chunk size and 100 chunk overlap
|   └── raw                                     # Raw data files (PDFs) used as the dataset for the experiments
├── notebooks                                   # Jupyter notebooks for data analysis
|   └── 01_exploring_results.ipynb              # Jupyter notebook for exploring the results
├── results                                     # Results files
|   └── results.csv                             # Results CSV file
├── src                                         # Source code
|   ├── configuration                           # Configuration models and validation
|   |   ├── configuration_model.py              # Configuration models
|   |   └── load_configuration.py               # Load configuration files script
|   ├── constants                               # Constants used in the project
|   |   ├── evaluation_config.py                # Evaluation configuration constants
|   |   ├── prompts.py                          # Prompt templates
|   |   └── questions.py                        # Questions for the experiments - evaluation dataset
|   ├── models                                  # Models for the RAG pipelines
|   |   ├── base_mixture_rag                    # Abstract base class for Mixture RAG
|   |   ├── base_simple_rag                     # Abstract base class for Simple RAG
|   |   ├── mixture_rag.py                      # Mixture RAG model
|   |   └── simple_rag.py                       # Simple RAG model
|   ├── pipelines                               # RAG pipelines
|   |   ├── mixture_rag_pipeline.py             # Mixture RAG pipeline
|   |   └── simple_rag_pipeline.py              # Simple RAG pipeline
|   └── utils                                   # Utility functions
|       └── evaluation_utils.py                 # Evaluation utility functions
├── .env                                        # Environment variables file (not included in the repository)
├── .gitignore                                  # Git ignore file
├── docker-compose.yml                          # Docker Compose file for Langfuse Server
├── execute_pipeline.py                         # Script to execute the RAG pipeline
├── extract_results.py                          # Script to extract the results
├── README.md                                   # README.md file (this file)
└── requirements.txt                            # Python requirements file
```
