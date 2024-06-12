import argparse
import text
from tqdm import tqdm
from utils import load_filepaths_and_text

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--out_extension", default="cleaned")
  parser.add_argument("--text_index", default=1, type=int)
  parser.add_argument("--filelists", nargs="+", default=["data/custom_dataset/train.csv", "data/custom_dataset/valid.csv"]) # train and valid files created with prepare_data.py
  parser.add_argument("--text_cleaners", nargs="+", default=["portuguese_cleaners"]) # portuguese_cleaners added at cleaners.py

  args = parser.parse_args()
  vocab = ''
   
  for filelist in args.filelists:
    print("START:", filelist)
    filepaths_and_text = load_filepaths_and_text(filelist)
    for i in tqdm(range(len(filepaths_and_text))):
      original_text = filepaths_and_text[i][args.text_index]
      cleaned_text = text._clean_text(original_text, args.text_cleaners)
      # Do this to confirm that all symbols in the dataset are in symbols (vocabulary model)
      vocab = list(set(cleaned_text) | set(vocab))
      filepaths_and_text[i][args.text_index] = cleaned_text

    new_filelist = filelist + "." + args.out_extension
    with open(new_filelist, "w", encoding="utf-8") as f:
      f.writelines(["|".join(x) + "\n" for x in filepaths_and_text])

  print(f"Dataset vocabulary: {''.join(vocab)}")

  new_symbols = []
  for symbol in vocab:
      if symbol not in text.symbols:
          new_symbols.append(symbol)

  if len(new_symbols) > 0:
    print(f"The following symbols in the dataset are not in the vocabulary: {new_symbols}")
  else:
    print("All symbols in the dataset are in the vocabulary!")