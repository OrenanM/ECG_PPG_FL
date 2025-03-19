# PFLLib com Dados Biomédicos

Este repositório estende a biblioteca PFLLib tradicional, adicionando suporte para dados biomédicos. 

## Principais Modificações
As principais mudanças incluem a inserção de dados biomédicos, que podem ser gerados pelo arquivo `generate_BioSIGNAL.py` localizado na pasta `dataset`.

Os argumentos disponíveis para a geração de dados são:

- `niid`: Determina se os dados gerados serão Non-IID.
- `balance`: Define se os dados serão balanceados.
- `partition`: Especifica o tipo de particionamento.
- `size_win`: Define o tamanho da janela (Obs: 1250 foi utilizado em nossas simulações).
- `hold_size`: Determina se há sobreposição.
- `type_signal`: Define o tipo de dado biomédico.

### Tipos de `type_signal` Suportados

- **`ECG` ou `PPG`** → Geração de dados individualmente.
- **`ECG_PPG`** → Geração de dados em conjunto.
- **`Fusion`** → Geração de dados fundidos posteriormente.

## Executando Simulações

Para iniciar as simulações, execute o arquivo `main.py`, utilizando o parâmetro `-data` para especificar o tipo de dados e a abordagem a ser utilizada.

### Exemplos de Execução
```bash
python main.py -nc 40 -gr 300 -data Fusion -t 50
python main.py -nc 40 -gr 300 -data ECG -t 50 -algo FedAvg
```

As linhas acima executam, respectivamente:
- Algoritmo com dados de fusão.
- Algoritmo utilizando apenas dados de ECG com `FedAvg`.

### Configuração de Envio Parcial da Rede

Para configurar o envio parcial da rede (split), utilize o argumento `-algo` com o valor `FedAvgBio`.

```bash
python main.py -nc 40 -gr 300 -data Fusion -t 50 -algo FedAvgBio
```

Esse comando executará nosso algoritmo com fusão de dados e envio parcial da rede.