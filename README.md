## Readme

Python 3.9.1

## Conteúdo:

 - candle5m.csv (todas candles de 5 minutos já formatadas para uso. Possivel abrir no Excel)
 - data5m.txt (Dado cru dos candles recebidas pela Binance, em json)
 - growing.py (Teste de "interface", com candles moveis)
 - main.py (Conecta na Binance e baixa os dados no arquivo data5m.txt)
 - parser.py (Formata os dados do arquivo data5m.py -> candle5m.csv)
 - websocketsFUT.py (Conecta e baixa os dados de 5m minutos do "indice futuro")
 - obv.py (Calculo do indicador OBV, de acordo com [Investopedia](https://www.investopedia.com/terms/o/onbalancevolume.asp)

## Guia para instalacao das bibliotecas no Windows

 - **pip install pandas numpy websocketclient json matplotlib mplfinance**

O Tensorflow é possivel ser instalado pela ferramenta pip ou baixar no site.

## TODO

 - Recomendacoes de referencias (livros, links, etc)
 - o proprio **TODO**
 

