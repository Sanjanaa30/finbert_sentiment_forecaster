import pandas as pd

def main():
    path = "data/raw/analyst_ratings_processed.csv"
    df = pd.read_csv(path)
    print(df.head())
    print("rows:", len(df), "cols:", df.columns.tolist())

if __name__ == "__main__":
    main()