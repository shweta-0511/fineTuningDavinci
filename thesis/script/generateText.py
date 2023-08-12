print("Importing the libraries")
import pandas as pd
import openai
import re
from bs4 import BeautifulSoup
import numpy as np
import time

print("Defining column names for dataframe")
columns = ['terminologyEnglish', 'terminologyFrench']

print("Create an empty dataframe with defined columns")
df = pd.DataFrame(columns=columns)

#Setting up your OpenAI API credential
openai.api_key = 'your key'

print("Reading the SGM file")
with open("/home/ubuntu/thesis/data/input/test.en-fr.fr.sgm", "r") as file:
    sgm_content = file.read()

# Parse the SGM content using BeautifulSoup
soup = BeautifulSoup(sgm_content, "html.parser")

# Find all <seg> tags
seg_tags = soup.find_all("seg")

# Extract the values of src and tgt attributes from each <seg> tag
data = []
# Extract the value of the src attribute for each <seg> tag
for seg_tag in seg_tags:
    term_tags = seg_tag.find_all("term")
    for term_tag in term_tags:
        src_value = term_tag.get("src")
        tgt_value = term_tag.get("tgt")
        data.append({"src_value": src_value, "tgt_value": tgt_value})

# Add src_value and tgt_value to df columns 'terminologyEnglish' and 'terminologyFrench'
for i, row in enumerate(data):
    terminologyEnglish = row["src_value"]
    terminologyFrench = row["tgt_value"]
    df.loc[i] = [terminologyEnglish, terminologyFrench]

print("Dropping duplicate entries from dataframe")
df = df.drop_duplicates()
df = df.reset_index(drop=True)

# Define your input prompts
prompts = df['terminologyEnglish']

dfs = []
i = 0

print("Function to generate sentences for a given prompt")
def generate_sentences(prompt):
  responses = []
  for _ in range(5):
    try:
      response = openai.Completion.create(
        engine="text-davinci-003",
        prompt="Generate an English sentence for this term: " + prompt,
        max_tokens=50,
        n=1,
        stop=None,
        temperature=0.8,
      )
      generated_text = response.choices[0].text.strip()
      responses.append(generated_text)
      time.sleep(3)
    except Exception as e:
      print(f"Error occurred while making API call: {str(e)}")
  return responses

print("Create a new DataFrame to store generate sentences")
df_generated = pd.DataFrame(columns=['terminologyEnglish', 'syntheticText'])

try:
  for index, row in df.iterrows():
    term = row['terminologyEnglish']
    print("Generating synthetic text for term :"+term)
    generated_sentences = generate_sentences(term)
    for sentence in generated_sentences:
      df_generated = pd.concat([df_generated, pd.DataFrame({'terminologyEnglish': [term], 'syntheticText': [sentence]})], ignore_index=True)
      
  # Merge the generated sentences DataFrame with the original DataFrame
  df = pd.merge(df, df_generated, on='terminologyEnglish')
except Exception as e:
  print("Code failed with error: "+str(e))

print("Generating translated statements")

try:
    for _, row in df.iterrows():
        terminology = row['terminologyEnglish']
        syntheticText = row['syntheticText']
        
        if pd.isna(syntheticText):
            continue  # Skip iteration if syntheticText is NaN
        
        prompt = "Translate the following English text to French: "+syntheticText
        # Generate translation using the OpenAI API
        try:
            response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=50,
            n=1,
            stop=None,
            )
            time.sleep(3)
        except Exception as e:
            print(f"Error occurred while making API call: {str(e)}")
            time.sleep(5)
        translated_text = response.choices[0].text.strip()
        df.loc[df.index == _, 'translatedText'] = translated_text
        print("Translated sentence for term: "+terminology)
        
except Exception as e:
    print("Code failed with error: "+str(e))

df.to_csv("/home/ubuntu/thesis/data/output/translatedData.csv")
