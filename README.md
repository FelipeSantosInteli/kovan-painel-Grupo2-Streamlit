# kovan-painel-Grupo2-Streamlit

Para rodar localmente a aplicação do streamlit, entre na pasta e, por meio de um terminal, instale as bibliotecas necessárias por meio do requirements.txt

___

### Virtual Envirorment Python - venv (opcional)

É possível iniciar um Ambiente Virtual Python para instalação dos requirements

Para isso, na pasta do projeto, utilize o seguinte comando no terminal

```bash
python -m venv venv
```

Uma pasta chamada _venv_ irá aparecer no diretório. Nela (geralmente em /bin ou /Scripts) terão alguns arquivos com o nome **activate**, cada um de um tipo. Execute no terminal o arquivo correspondente ao seu tipo de terminal (.ps1 para _powershell_, .fish para _fish_, etc) apenas escrevendo o nome/caminho e apertando Enter. Exemplos:

- ./venv/bin/activate.ps1

- ./venv/Script/activate.fish


- activate.ps1 (caso já esteja dentro de ./venv/bin)


Se tudo der certo, na linha de execução do seu terminal aparecerá a palavra _venv_ indicando que ele está sendo executado dentro do ambiente virtual.

___

### Instalando requirements

No terminal aberto dentro da pasta, utilize o seguinte comando para instalar todas as dependências:

```bash
pip install -r requirements.txt
```

### Rodando a aplicação localmente

Com as dependências instaladas, utilize o seguinte comando para iniciar a aplicação:

```bash
streamlit run app_dashboard_agentes_v2.py
```

Para encerrar a aplicação vá para o terminal no qual ela foi iniciada (onde o comando anterior foi executado) e aperte <ctrl+C>.

O painel também pode ser acessado por meio do deploy do próprio Streamlit nesse [link](https://kovan-painel-grupo2-app.streamlit.app/).
