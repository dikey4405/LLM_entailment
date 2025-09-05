import pandas as pd


def load_generated_answers(file_path: str, col_name="generated_response"):
    df = pd.read_csv(file_path)
    return df[col_name].tolist()


def load_knowledge(file_path: str, col_name="context"):
    df = pd.read_csv(file_path)
    return df[col_name].tolist()
