import pandas as pd
import json

data = pd.read_csv('/home/ubuntu/thesis/data/output/translatedData.csv')

df = data[['syntheticText','translatedText']]

df.rename(columns={'syntheticText': 'prompt', 'translatedText': 'completion'}, inplace=True)

df.to_json('/home/ubuntu/thesis/fineTune/data/data.json', orient='records', lines=True)
