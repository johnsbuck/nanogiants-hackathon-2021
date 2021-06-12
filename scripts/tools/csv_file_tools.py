import csv


def get_csv_data(file, row_checker=lambda x: True):
    """

    Args:
        file (str): File path for opening the csv file.
        row_checker (function): Function that returns True if row is valid, otherwise False.
            Default function that returns True.

    Returns:
        List[List]: A list containing each row from the CSV file.
    """
    data = []

    with open(file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row_checker(row):
                continue
            data.append(row)
    return data
