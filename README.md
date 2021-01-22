## Readme

 - Python 3.9.1
 - Tensorflow 2.4
 - Git 2.30

## Conteúdo:

 - candle5m.csv (todas candles de 5 minutos já formatadas para uso. Possível abrir no Excel)
 - data5m.txt (dado cru dos candles recebidas pela Binance, em json)
 - growing.py (teste de "interface", com candles móveis)
 - main.py (conecta na Binance e baixa os dados no arquivo data5m.txt)
 - parser.py (formata os dados do arquivo data5m.txt -> candle5m.csv)
 - websocketsFUT.py (conecta e baixa os dados de 5 minutos do "índice futuro")
 - obv.py (cálculo do indicador OBV, de acordo com [Investopedia](https://www.investopedia.com/terms/o/onbalancevolume.asp))

## Guia para instalação das bibliotecas no Windows

 - Para rodar o python pelo CMD ou preferivelmente, pelo PowerShell, é necessário marcar na tela inicial de instalação "Add Python 3.9 to PATH"
 - Bibliotecas: **pip install pandas numpy websocketclient matplotlib mplfinance**

O Tensorflow é possivel ser instalado pela ferramenta pip ou baixar no site.

## TODO

 - Recomendaçoes de referências (livros, links, etc)
 - o próprio **TODO**
 

