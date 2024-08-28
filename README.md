# small-language-model-rags-is-all-you-need

This is a repository that contains the code for the experiments conducted for the project called "Small Language Model RAGs is all you need". The project aims to showcase the capabilities of building RAG systems on top of smaller language models like Gemma2 9B, Llama 3 8B, Mistral 7B, and others. The idea of the project is how we can leverage the capabilities of smaller language models with some smart prompt engineering and some interesting inovative ideas to have results that are comparable or in some cases better than RAG systems built on top of larger language models like GPT-4o, Claude 3.5 Sonnet, and others.

For the experiments we used two different approaches or with other words types of RAG systems:
- **Simple RAG** - The classic RAG system pipeline
- **Mixture of RAG** (Mixture RAG) - RAG system pipeline inspired by the implementation of Mixture of Agents (MoA)

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
- **Simple RAG** - The classic RAG system pipeline
- **Mixture of RAG (Mixture RAG)** - RAG system pipeline inspired by the implementation of Mixture of Agents (MoA)

The example configuration for the Simple RAG (can be found in `config/simple.rag.example.yaml`) looks like:
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

The example configuration for the Mixture RAG (can be found in `config/mixture.rag.example.yaml`) looks like:
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

The validation of the configuration files is done with Pydantic (more info about it can be seen in the configuration models `src/configuration/configuration_model.py`)

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
2. Install the required Python packages with the following command:
```bash
pip install -r requirements.txt
```
3. Create a `.env` file in the root directory of the project and add the environment variables(see the Environment Variables section)
4. Create a configuration file for the experiment you want to run(see the Configuration for experiments section) and put in the `src/configuration` folder
5. Setup the Langfuse Server:
    - 4a. Use their Cloud Service hosted Langfuse
    - 4b. Use the Docker Compose file to run the Langfuse Server locally
6. Modify the `execute_pipeline.py` file with the configuration file you want to run and the prompt templates you want to use.
7. Run the `execute_pipeline.py` file with the following command:
```bash
python execute_pipeline.py
```
8. Generate the results by running the `extract_results.py` file with the following command:
```bash
python extract_results.py
```
9. The results will be saved in the `results` folder and you can analyze them.


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

![image](https://github.com/user-attachments/assets/ee34da3d-6be8-4b92-9943-36abedfc575a)

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

![image](https://github.com/user-attachments/assets/649467b8-bafa-4d85-831a-dab052314662)


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

For the experiments, the results are stored in a CSV file in the `results` folder. Those results are extracted from the Langfuse Server which contain detailed traces and metrics for each experiment. The results are extracted using the `extract_results.py` script.

#### Metrics

The metrics used for the evaluation of the experiments are:

- **Faithfulness**: This measures the factual consistency of the generated answer against the given context. It is calculated from answer and retrieved context. The answer is scaled to (0,1) range. Higher the better. The generated answer is regarded as faithful if all the claims made in the answer can be inferred from the given context. To calculate this, a set of claims from the generated answer is first identified. Then each of these claims is cross-checked with the given context to determine if it can be inferred from the context.

<div style = "text-align: center;">
  <p>
    <span style="display: inline-block; vertical-align: middle;">
      <img src="https://latex.codecogs.com/png.latex?\text{Faithfullness}=\frac{\text{Number%20of%20claims%20in%20the%20generated%20answer%20that%20can%20be%20inferred%20from%20given%20context}}{\text{Total%20number%20of%20claims%20in%20the%20generated%20answer}}" alt="Faithfulness Formula">
    </span>
  </p>
</div>


- **Answer Relevancy**: The evaluation metric, Answer Relevancy, focuses on assessing how pertinent the generated answer is to the given prompt. A lower score is assigned to answers that are incomplete or contain redundant information and higher scores indicate better relevancy. This metric is computed using the question, the context and the answer. The Answer Relevancy is defined as the mean cosine similarity of the original question to a number of artifical questions, which where generated (reverse engineered) based on the answer:

<div style="text-align: center;">
  <p>
    <span style="display: inline-block; vertical-align: middle;">
      <img src="https://latex.codecogs.com/png.latex?\text{Answer%20Relevancy}%20=%20\frac{1}{N}%20\sum_{i=1}^{N}%20\cos(E_{g_i},%20E_o)" alt="Answer Relevancy Formula 1">
    </span>
  </p>

  <p>
    <span style="display: inline-block; vertical-align: middle;">
      <img src="https://latex.codecogs.com/png.latex?\text{Answer%20Relevancy}%20=%20\frac{1}{N}%20\sum_{i=1}^{N}%20\frac{E_{g_i}%20\cdot%20E_o}{\|E_{g_i}\|\%20\|E_o\|}" alt="Answer Relevancy Formula 2">
    </span>
  </p>
</div>

Where:
<div>
  <p>
    - <span style="display: inline-block; vertical-align: middle;">
      <img src="https://latex.codecogs.com/png.latex?E_{g_i}" alt="E_{g_i}"> is the embedding of the <img src="https://latex.codecogs.com/png.latex?i^{\text{th}}" alt="i^{th}"> artificial question generated from the answer.
    </span>
  </p>

  <p>
    - <span style="display: inline-block; vertical-align: middle;">
      <img src="https://latex.codecogs.com/png.latex?E_o" alt="E_o"> is the embedding of the original question.
    </span>
  </p>

  <p>
    - <span style="display: inline-block; vertical-align: middle;">
      <img src="https://latex.codecogs.com/png.latex?N" alt="N"> is the number of artificial questions generated from the answer.
    </span>
  </p>
</div>

> [!NOTE]
> Eventhough in practice the score will range between 0 and 1 most of the time, this is not mathematically guranteed, due to the nature of the cosine similarity ranging from -1 to 1.


- **Context Utilization**: Context utilization measures the extent to which the retrieved context aligns with the annotated answer, treated as the ground truth. It is computed using question, ground truth and the retrieved context, and the values range between 0 and 1, with higher values indicating better performance. To estimate context utilization from the ground truth answer, each claim in the ground truth answer is analyzed to determine whether it can be attributed to the retrieved context or not. In an ideal scenario, all claims in the ground truth answer should be attributable to the retrieved context. If the ground truth is not provided, the judge evaluator LLM is used to generate the ground truth answer.

<div style="text-align: center;">
  <p>
    <span style="display: inline-block; vertical-align: middle;">
      <img src="https://latex.codecogs.com/svg.latex?\text{Context%20Utilization}%20=%20\frac{\text{GT%20claims%20that%20can%20be%20attributed%20to%20context}}{\text{Number%20of%20claims%20in%20GT}}" alt="Context Utilization Formula">
    </span>
  </p>
</div>

#### Judge LLM and Embedder
For Judge LLM Evaluator it is worked with Claude 3.5 Sonnet model and the Amazon Titan Embed Text 2 model with 512 dimensions. The configuration for the Judge LLM and Embedder can be found in the `src/constants/evaluation_config.py` file.


#### Analysis of the Results


The initial exploration of the results focused on identifying problematic questions, specifically those with lower scores. The objective was to refine the experiments by excluding these less effective questions and concentrating on the 10 most relevant ones. This approach aims to enhance the overall quality and reliability of the experiments by ensuring that only the most pertinent questions and answers are considered.

To identify these problematic questions, the dataset was grouped by individual questions. For each question, the mean scores were calculated across three key metrics: faithfulness, answer relevancy, and context utilization. These mean scores provided a comprehensive view of each question's performance. Subsequently, an overall average score was computed for each question by taking the basic average of the mean scores from the three metrics. This overall score was then used to rank the questions, allowing for an informed decision on which questions to exclude from the experiments.

<h4>Questions with the lowest scores</h4>
<table>
  <thead>
    <tr>
      <th>Index</th>
      <th>Question</th>
      <th>Score</th>
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


From the table, we can observe which questions have the lowest scores. Specifically, the last four questions exhibit the lowest performance and are therefore excluded from the subsequent analysis. This exclusion helps to focus the analysis on the more reliable and relevant questions, ensuring that the results are not skewed by outliers or less effective queries.

The next step involves a detailed analysis of the results for each experiment. This analysis includes ranking the experiments based on the average scores for each metric: faithfulness, answer relevancy, and context utilization. For clarity and comprehensiveness, the top 14 experiments for each metric are highlighted and presented below. Additionally, an overall ranking is conducted by calculating the average of the average scores across all metrics. This comprehensive ranking provides a holistic view of the experiments' performance, facilitating a more informed evaluation and comparison.

<h4>Faithfulness</h4>
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
    <tr><td>13</td><td>simple-rag-claude-3-opus</td><td>0.867106</td></tr>
    <tr><td>14</td><td>simple-rag-gpt-4o-mini</td><td>0.851786</td></tr>
  </tbody>
</table>

The table above ranks various experiments based on their faithfulness scores, which measure how accurately the generated responses adhere to the source information. Based on the results from the table, it is evident that the scores of the RAG  systems based on smaller language models are very close to, or in some cases even better than, those based on larger language models. For instance, in the top 7 scores, we have 4 RAG systems that are based on smaller language models: `simple-rag-llama-3.1-8b`, `mixture-rag-gemma2-9b-it-thought` - which is a combination of multiple smaller language, `simple-rag-gemma-7b-it`, and `simple-rag-llama-3-8b`. These smaller models achieve faithfulness scores of 0.957778, 0.924542, 0.923677, and 0.913214 respectively, which are comparable to or even surpass the scores of some larger models. 

This observation suggests that smaller language models can perform nearly as well as, or sometimes better than, larger models in terms of faithfulness. The close scores among the top experiments indicate that model architecture and training strategies play a significant role in achieving high faithfulness, regardless of the model size. This insight is valuable for guiding future improvements and optimizations in model development, as it highlights the potential of smaller models to deliver high-quality results, results that are faithful to the context and source information provided.

<h4>Answer Relevancy</h4>
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
    <tr><td>14</td><td>simple-rag-claude-3-haiku</td><td>0.865661</td></tr>
  </tbody>
</table>

The table above ranks various experiments based on their answer relevancy scores, which measure the relevance of the generated responses to the given prompts. The results show that in the top 7 experiments, 4 of them are again based on smaller language models with simple rag pipeline approach or with the smart technique of mixture rag pipeline approach. The experiments `simple-rag-mistral-7b-instruct`, `mixture-rag-gemma2-9b-it-thought`, `simple-rag-gemma2-9b-it` and `mixture-rag-llama3.1-8b-instruct-thought` have really high answer relevancy scores of 0.914597, 0.910476, 0.905305, and 0.897726 respectively. 

This again indicates that smaller language models can generate highly relevant responses that are closely aligned with the given prompts. We can even see that the mixture rag pipeline approach with the smart technique of choosing the best response from the generated responses(thought) can achieve high answer relevancy scores. 


<h4>Context Utilization</h4>
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
    <tr><td>9</td><td>simple-rag-gpt-4o</td><td>0.897222</td></tr>
    <tr><td>10</td><td>mixture-rag-gemma2-9b-it-modified</td><td>0.880556</td></tr>
    <tr><td>11</td><td>mixture-rag-gemma2-9b-it-thought</td><td>0.880556</td></tr>
    <tr><td>12</td><td>simple-rag-llama-3.1-8b</td><td>0.880556</td></tr>
    <tr><td>13</td><td>simple-rag-gemma-7b-it</td><td>0.875000</td></tr>
    <tr><td>14</td><td>simple-rag-llama-3-8b</td><td>0.875000</td></tr>
  </tbody>
</table>

The table above ranks various experiments based on their context utilization scores, which measure how effectively the retrieved context aligns with the annotated answers. Here really we can see how RAG systems based on smaller language models are performing really well in terms of context utilization. From the best 14 experiments, 11 of them are based on smaller language models. Another interesting thing is that mixture RAG approaches are excellent in context utilization, with 3 of the top 5 experiments being based on the mixture RAG approach. The experiments `mixture-rag-llama3.1-8b-instruct`, `mixture-rag-mixtral-8x7-instruct-modified`, and `mixture-rag-mixtral-8x7-instruct` have context utilization scores of 0.916667, 0.916667, and 0.913889 respectively.

<h4>Average of the Average Scores</h4>
<table>
  <thead>
    <tr>
      <th>Index</th>
      <th>Experiment</th>
      <th>Average Score</th>
    </tr>
  </thead>
  <tbody>
    <tr><td>1</td><td>mixture-rag-gemma2-9b-it-thought</td><td>0.905191</td></tr>
    <tr><td>2</td><td>simple-rag-mistral-7b-instruct</td><td>0.900319</td></tr>
    <tr><td>3</td><td>simple-rag-llama-3.1-405b-instruct</td><td>0.896580</td></tr>
    <tr><td>4</td><td>simple-rag-mixtral-8x7b-instruct</td><td>0.896383</td></tr>
    <tr><td>5</td><td>simple-rag-gpt-4o</td><td>0.892935</td></tr>
    <tr><td>6</td><td>simple-rag-gpt-4o-mini</td><td>0.890044</td></tr>
    <tr><td>7</td><td>simple-rag-llama-3.1-70b-instruct</td><td>0.890022</td></tr>
    <tr><td>8</td><td>simple-rag-gemma-7b-it</td><td>0.887449</td></tr>
    <tr><td>9</td><td>simple-rag-llama-3.1-8b</td><td>0.887003</td></tr>
    <tr><td>10</td><td>mixture-rag-mixtral-8x7-instruct-modified</td><td>0.886127</td></tr>
    <tr><td>11</td><td>simple-rag-llama-3-70b</td><td>0.883451</td></tr>
    <tr><td>12</td><td>simple-rag-llama-3-8b</td><td>0.881460</td></tr>
    <tr><td>13</td><td>simple-rag-gemma2-9b-it</td><td>0.871802</td></tr>
    <tr><td>14</td><td>mixture-rag-gemma2-9b-it-modified</td><td>0.857831</td></tr>
  </tbody>
</table>

The table above ranks various experiments based on their average scores, which provide a comprehensive view of the experiments' performance across all metrics. The results show the dominance of RAG systems based on smaller language models, with 9 of the top 14 experiments being based on smaller models.


#### Conclussion

The analysis of various experiments comparing RAG (Retrieval-Augmented Generation) systems based on different language models yields several significant insights:

- Smaller language models perform competitively when used in RAG system, often achieving scores comparable to or even surpassing those of larger language models based RAG systems across multiple metrics (faithfulness, answer relevancy, and context utilization).

- The mixture RAG pipeline where the generator of the RAG system is inspired by the implementation of Mixture of Agents(MoA) technique like choosing the best response from generated output options, shows strong performance across metrics.

- The close scores among top experiments suggest that factors such as model architecture and training strategies may be more crucial than model size in achieving high-quality results.

- Smaller models and mixture RAG approaches demonstrate particular effectiveness in context utilization, indicating their ability to align retrieved information with annotated answers.

- Overall when considering average scores across all metrics, RAG systems based on smaller language models dominate the top rankings, occupying 9 out of the top 14 positions.

These findings highlight the potential of smaller language models and sophisticated RAG approaches to deliver high-quality, faithful, and relevant responses while efficiently utilizing context. 

Moreover, we do not need to work the additional benefits of the smaller language models, such as:

- Self-hosting and open-source capabilities: Smaller models are more amenable to self-hosting, allowing organizations to maintain control over their data and infrastructure. Many of these models are also open-source, fostering transparency, community-driven improvements, and customization.

- Improved efficiency and reduced costs: Smaller models require less computational resources, leading to lower energy consumption and reduced operational costs. This efficiency makes them more environmentally friendly and economically viable for a broader range of applications.

- Democratization of AI: The ability to run these models on less powerful hardware democratizes access to advanced AI capabilities. This allows individuals, small businesses, and organizations with limited resources to create and deploy sophisticated RAG systems, fostering innovation across diverse sectors.

- Faster inference times: Smaller models typically offer quicker response times, which is crucial for real-time applications and enhancing user experience in interactive systems.

- Privacy and compliance advantages: Self-hosted smaller models can help organizations better comply with data protection regulations and maintain stricter control over sensitive information.

- Flexibility and adaptability: Smaller models are often easier to fine-tune or adapt to specific domains or tasks, allowing for more tailored solutions without the need for extensive computational resources.

These insights and benefits could guide future developments in language model applications, potentially leading to more resource-efficient, accessible, and equally effective AI systems. By leveraging smaller language models in RAG systems, organizations and individuals can harness powerful AI capabilities while enjoying greater flexibility, control, and cost-effectiveness.


## Project Structure:
```
.
├── .github                                     # GitHub Actions workflows
|   └── workflows                               # Workflows files
|       └── python-formating.yml                # Python formatting workflow with Ruff
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
