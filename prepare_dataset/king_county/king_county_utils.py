import pandas as pd


def make_transform(frame):
    """
    Transform some numerical features to categorical.
    :type frame: pd.DataFrame
    :param frame:
    """
    # Some numerical features are actually really categories
    frame.replace(
        {
            "waterfront": {0: "No", 1: "Yes"}
        },
        inplace=True
    )


def parse_date(frame):
    """
    Transform some numerical features to categorical.
    :type frame: pd.DataFrame
    :param frame:
    """
    frame["year_sold"] = frame["date"].apply(lambda row: int(row[:4]))
    frame["month_sold"] = frame["date"].apply(lambda row: int(row[4:6]))
    frame.replace(
        {
            "month_sold": {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                           7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
        },
        inplace=True
    )
