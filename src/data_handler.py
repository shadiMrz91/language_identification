import os
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_dataset


class DataHandler:
    def __init__(self, raw_path="data/raw", splits_path="data/splits",
                 file_name="dataset.csv"):
        self.raw_path = raw_path
        self.splits_path = splits_path
        self.file_name = file_name

        os.makedirs(self.raw_path, exist_ok=True)
        os.makedirs(self.splits_path, exist_ok=True)

        self.train_df = None
        self.dev_df = None
        self.test_df = None
        self.data_loaded = False

    def _download_dataset(self, languages=None, max_samples=1000):
        if languages is None:
            languages = ["es", "fr", "en", "it", "pt", "nl", "sv", "pl", "ru", "ja"]

        dataset_path = os.path.join(self.raw_path, self.file_name)
        print(" Downloading dataset from Hugging Face...")
        all_rows = [] 

        for lang in languages:
            try:
                print(f" Loading data for language: {lang}")
                dataset = load_dataset("tatoeba", lang1="de", lang2=lang, trust_remote_code=True)
                sentences = dataset["train"]["translation"]
                df = pd.DataFrame(sentences)

                df = df.head(max_samples)
                temp = pd.DataFrame({
                    "text": df[lang],
                    "lang": lang
                }).dropna()
                temp = temp[temp["text"].str.strip() != ""]
                all_rows.append(temp)
                print(f"  {len(temp)} samples added for language {lang}")

            except Exception as e:
                print(f" Error loading {lang}: {e}")

        if not all_rows:
            raise RuntimeError(" No data downloaded from any language.")

        final_df = pd.concat(all_rows, ignore_index=True)
        final_df.to_csv(dataset_path, index=False)
        print(f" Dataset saved at {dataset_path} with {len(final_df)} samples.")
        return final_df

    def _create_splits(self, df, ratios=(0.7, 0.15, 0.15)):
        print(" Creating new train/dev/test splits...")

        train_df, temp_df = train_test_split(df, test_size=round(1 - ratios[0],2),
                                             stratify=df["lang"], random_state=42)
        dev_df, test_df = train_test_split(temp_df,
                                           test_size=ratios[2] / (ratios[1] + ratios[2]),
                                           stratify=temp_df["lang"], random_state=42)

        train_df.to_csv(os.path.join(self.splits_path, "train.csv"), index=False)
        dev_df.to_csv(os.path.join(self.splits_path, "dev.csv"), index=False)
        test_df.to_csv(os.path.join(self.splits_path, "test.csv"), index=False)

        print(f" Splits created: Train={len(train_df)}, Dev={len(dev_df)}, Test={len(test_df)}")

    def _load_splits(self):
        print(" Loading split files...")
        train_path = os.path.join(self.splits_path, "train.csv")
        dev_path = os.path.join(self.splits_path, "dev.csv")
        test_path = os.path.join(self.splits_path, "test.csv")

        if not all(os.path.exists(p) for p in [train_path, dev_path, test_path]):
            raise FileNotFoundError("Split files not found. Run prepare_data to create them.")

        self.train_df = pd.read_csv(train_path)
        self.dev_df = pd.read_csv(dev_path)
        self.test_df = pd.read_csv(test_path)
        self.data_loaded = True
        print(" Data successfully loaded from split files.")


    def _is_splits_ready(self):
        return all(os.path.exists(os.path.join(self.splits_path, f))
                   for f in ["train.csv", "dev.csv", "test.csv"])

    def _is_raw_ready(self):
        return os.path.exists(os.path.join(self.raw_path, self.file_name))

    def prepare_data(self, ratios=(0.7, 0.15, 0.15)):      
        print(" Starting data preparation pipeline...")

        if self._is_splits_ready():
            print(" Found existing splits. Loading them...")
            self._load_splits()

        elif self._is_raw_ready():
            print(" Raw dataset found, creating splits...")
            df = pd.read_csv(os.path.join(self.raw_path, self.file_name))
            self._create_splits(df, ratios)
            self._load_splits()

        else:
            print(" No data found. Downloading...")
            df = self._download_dataset()
            self._create_splits(df, ratios)
            self._load_splits()

        print(" Data preparation complete.")
        return self._get_data()

    def _get_data(self):
        if not self.data_loaded:
            raise ValueError("Data not loaded yet. Call prepare_data() first.")
        return {
            "train": (self.train_df["text"].tolist(), self.train_df["lang"].tolist()),
            "dev": (self.dev_df["text"].tolist(), self.dev_df["lang"].tolist()),
            "test": (self.test_df["text"].tolist(), self.test_df["lang"].tolist()),
        }

  