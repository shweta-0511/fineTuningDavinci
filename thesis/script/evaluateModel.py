import sacrebleu
import pandas as pd

df = pd.read_table('test.en-fr.tsv')
dfModel = pd.read_csv('"/home/ubuntu/thesis/data/output/modelOutput.csv"')
# Define the reference translations
references = df['targetString'].tolist()

dfModel['frenchTextDavinciFineTuned'] = dfModel['frenchTextDavinciFineTuned'].astype(str)
dfModel['frenchTextDavinci'] = dfModel['frenchTextDavinci'].astype(str)
dfModel['frenchTextDavinci002'] = dfModel['frenchTextDavinci002'].astype(str)

# Define the translated text
translationFineTune = dfModel['frenchTextDavinciFineTuned'].tolist()
translationDavinci = dfModel['frenchTextDavinci'].tolist()
translationDavinci002 = dfModel['frenchTextDavinci002'].tolist()

# Calculate the SacreBLEU score
bleuFineTuned = sacrebleu.corpus_bleu(translationFineTune, [references])
bleuDavinci = sacrebleu.corpus_bleu(translationDavinci, [references])
bleuDavinci002 = sacrebleu.corpus_bleu(translationDavinci002, [references])

print("Fine-Tuned model BLEU Score"+ str(bleuFineTuned.score))
print("Davinci model BLEU Score"+ str(bleuDavinci.score))
print("Davinci 002 model BLEU Score"+ str(bleuDavinci002.score))
