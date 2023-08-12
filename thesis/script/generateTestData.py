print("Importing the libraries")
import pandas as pd
import openai
import re
from bs4 import BeautifulSoup
import numpy as np
import time

print("Defining column names for dataframe")
columns = ['englishText', 'frenchTextDavinciFineTuned', 'frenchTextDavinci']

print("Create an empty dataframe with defined columns")
df = pd.DataFrame(columns=columns)

# Setting up your OpenAI API credentials
openai.api_key = 'your key'

 print("Reading the SGM file")
 with open("/home/ubuntu/thesis/data/input/blind_test.en-fr.en.sgm", "r") as file:
    sgm_content = file.read()

 # Parse the SGM content using BeautifulSoup
 soup = BeautifulSoup(sgm_content, "html.parser")

 # Find all <seg> tags
 seg_tags = soup.find_all("seg")

 # Extract the values of src and tgt attributes from each <seg> tag
 data = []
 # Extract the value of the src attribute for each <seg> tag
 for seg_tag in seg_tags:
   value = seg_tag.get_text().strip()
   data.append({"src_value": value})

 # Add src_value and tgt_value to df column 'englishText'
 for i, row in enumerate(data):
   englishText = row["src_value"]
   df.loc[i] = [englishText, None, None]
print("Generating translated statements using fine-tuned model")
cnt = 1
try:
    for _, row in df.iterrows():
        englishText = row['englishText']
        print(cnt)
        try:
            prompt = englishText
            # Generate translation using the OpenAI API
            response = openai.Completion.create(
            engine="your adapted model",
            prompt=prompt,
            max_tokens=50,
            n=1,
            stop=["\n"],
            )
            translated_text = response.choices[0].text.strip()
            print(translated_text)
            df.loc[df.index == _, 'frenchTextDavinciFineTuned'] = translated_text
            cnt = cnt+1
            time.sleep(3)
        except Exception as e:
            print("API call failed with error: "+str(e))
            time.sleep(5)
except Exception as e:
    print("Code failed with error: "+str(e))
print("Generating translated statements using davinci")

c =1
try:
    for _, row in df.iterrows():
        englishText = row['englishText']
        print(c)
        try:
            prompt = "Translate the following English text to French: "+englishText
            # Generate translation using the OpenAI API
            response = openai.Completion.create(
            engine="davinci",
            prompt=prompt,
            max_tokens=50,
            n=1,
            stop=None,
            )
            translated_text = response.choices[0].text.strip()
            df.loc[df.index == _, 'frenchTextDavinci'] = translated_text
            c=c+1
            time.sleep(3)
        except Exception as e:
            print("API call failed with error: "+str(e))
            time.sleep(5)
except Exception as e:
    print("Code failed with error: "+str(e))


print("Generating translated statements using text-davinci-002")

c =1
try:
    for _, row in df.iterrows():
        englishText = row['englishText']
        print(c)
        try:
            prompt = "Translate the following English text to French: "+englishText
            # Generate translation using the OpenAI API
            response = openai.Completion.create(
            engine="text-davinci-002",
            prompt=prompt,
            max_tokens=50,
            n=1,
            stop=None,
            )
            translated_text = response.choices[0].text.strip()
            df.loc[df.index == _, 'frenchTextDavinci002'] = translated_text
            c=c+1
            time.sleep(3)
        except Exception as e:
            print("API call failed with error: "+str(e))
            time.sleep(5)
except Exception as e:
    print("Code failed with error: "+str(e))

df.to_csv("/home/ubuntu/thesis/data/output/modelOutput.csv")

