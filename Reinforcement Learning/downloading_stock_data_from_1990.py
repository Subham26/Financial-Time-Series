from finrl import config_tickers
from finrl.finrl_meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.finrl_meta.data_processors.processor_yahoofinance import YahooFinanceProcessor

tickers = []
dp = YahooFinanceProcessor()
for tic in config_tickers.NAS_100_TICKER:
  try:
    df = dp.download_data(start_date = '1990-01-01',
                     end_date = '2022-03-19',
                     ticker_list = [tic], time_interval='1D')
    if len(df) == 8117:
      tickers.append(tic)
  except:
    print(tic)
print(tickers)
    
