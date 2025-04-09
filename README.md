# DocVQA-Data-Annotator

O repositório apresenta os códigos e prompts utilizados para o desenvolvimento da pesquisa exposta no artigo: A Data Annotation Approach Using Large Language Models. O repositório está estruturado da seguinte maneira:
- controller: código de controle para construção da representação textual do arquivo
- learning: código de controle das saídas dos modelos. Para os modelos executados localmente, gerencia a chamada e estrututuração da resposta.
- model: estruturas de dados utilizadas para armazenar as informações estruturadas pelos códigos de controle e dos modelos de aprendizado de máquina.
- prompts: pasta contendo os prompts utilizados para a etapa de geração das perguntas e respostas e para a avaliação automática dos gerados.
- samples: exemplos para execução do fluxo de anotação.
- utils: códigos utilatarios para auxiliar no funcionamento dos códigos de controle e para execução dos algoritmos de learning

