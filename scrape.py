from enum import Enum, auto
from typing import List
from dataclasses import dataclass
from datetime import datetime, date
from dateutil.rrule import rrule, MONTHLY
import dateutil.parser as dparser
import re
import asyncio
from pathlib import Path

from aiohttp import ClientSession, TCPConnector, ClientResponseError
import pandas as pd
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm_asyncio
from tenacity import retry, stop_after_attempt, wait_exponential_jitter
from fake_useragent import UserAgent


@dataclass
class Month:
    month: int
    year: int


URL = str


@dataclass
class CaseData:
    link: URL
    date: date
    judge: str
    decision: str


def generate_month_link(month: Month) -> URL:
    """
    Generates a link to court data for the given month.

    Args:
        month (Month): Month

    Returns:
        URL: Link to court data for the month.
    """
    url = f"https://nycourts.gov/reporter/motindex/mots_crimleav_{month.month:02}-{month.year}.htm"
    return url


def generate_links_for_time_range(start_month: Month, end_month: Month) -> List[URL]:
    """
    Generates a list of links to court data for each of the months in
    the given time range (all inclusive).

    Args:
        start_month (Month): Start month
        end_month (Month): End month

    Returns:
        List[URL]: List of links to court data
    """
    urls = []
    start_date = datetime(start_month.year, start_month.month, 1)
    end_date = datetime(end_month.year, end_month.month, 1)
    for dt in rrule(MONTHLY, dtstart=start_date, until=end_date):
        current_month = Month(dt.month, dt.year)
        month_link = generate_month_link(current_month)
        urls.append(month_link)
    return urls


# @retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(max=60))
async def get_case_links_from_month_link(
    session: ClientSession, month_link: URL
) -> List[URL]:
    """
    Get list of links to cases for each month.

    Args:
        session (ClientSession): The aiohttp request session.
        month (Month): Month

    Returns:
        List[str]: List of links to individual cases
    """
    headers = {"User-Agent": UserAgent().random}
    try:
        async with session.get(month_link, headers=headers) as response:
            response.raise_for_status()
            content = await response.read()
    except ClientResponseError as e:
        print(f"Request failed: {e}")
        raise

    soup = BeautifulSoup(content, features="html.parser")
    hyperlinks = soup.find_all("a")
    urls = [hyperlink.get("href") for hyperlink in hyperlinks]
    valid_urls = [
        url
        for url in urls
        if url is not None  # URL exists
        and url.startswith("http")  # Valid link
        and "nycourts.gov/reporter/motions" in url  # Is motion
    ]
    return valid_urls


# @retry(stop=stop_after_attempt(5), wait=wait_exponential_jitter(max=60))
async def parse_case_link(session: ClientSession, case_link: URL) -> CaseData:
    """
    Extract case data from a given case's link.

    Args:
        session (ClientSession): The aiohttp request session.
        case_link (URL): Case link.

    Returns:
        CaseData: Extracted case data.
    """
    headers = {"User-Agent": UserAgent().random}
    try:
        async with session.get(case_link, headers=headers) as response:
            response.raise_for_status()
            content = await response.read()
    except ClientResponseError as e:
        print(f"Request failed: {e}")
        raise

    soup = BeautifulSoup(content, features="html.parser")

    # Get date
    date_line = soup(string=re.compile("Decided on"))
    date_text = date_line[0][11:] if date_line else "No date"
    date = dparser.parse(date_text).date()

    # Get judge name
    judge_line = soup(string=re.compile("Judge: "))
    judge = judge_line[0][7:] if judge_line else "No judge"

    # Get decision
    decision_line = soup(string=re.compile("Disposition: "))
    decision = decision_line[0] if decision_line else "No decision"

    # Create and return final object
    case_data = CaseData(link=case_link, date=date, judge=judge, decision=decision)
    return case_data


CaseDataset = List[CaseData]


async def create_case_dataset_for_time_range(
    start_month: Month, end_month: Month
) -> CaseDataset:
    """
    Extract case data from all cases within a time range.

    Args:
        start_month (Month): Start month.
        end_month (Month): End month.

    Returns:
        CaseDataset: List of cases and their data.
    """
    time_range_links = generate_links_for_time_range(
        start_month=start_month, end_month=end_month
    )

    # Async extract all case links from time range
    async with ClientSession(connector=TCPConnector(limit=40)) as session:
        monthly_case_link_tasks = [
            get_case_links_from_month_link(session, month_link)
            for month_link in time_range_links
        ]
        monthly_case_link_results = await tqdm_asyncio.gather(*monthly_case_link_tasks)
        case_links = flatten(monthly_case_link_results)

        # Async extract data from each case
        case_data_tasks = [
            parse_case_link(session, case_link) for case_link in case_links
        ]
        case_data_results = await tqdm_asyncio.gather(*case_data_tasks)

    return case_data_results


def flatten(l: List) -> List:
    """Flatten list"""
    return [element for sublist in l for element in sublist]


def convert_case_dataset_to_df(case_dataset: CaseDataset) -> pd.DataFrame:
    """
    Convert case dataset to a Pandas dataframe.

    Args:
        case_dataset (CaseDataset): Case data.

    Returns:
        pd.DataFrame: Dataframe with case data.
    """
    return pd.DataFrame([vars(case_data) for case_data in case_dataset])


def get_all_cases(start_month: Month, end_month: Month, path: Path) -> pd.DataFrame:
    """
    Get all case data from the given time range and save as a CSV.

    Args:
        start_month (Month): Starting month of time range.
        end_month (Month): Ending month of time range.
        path (str): Filename to save the resulting dataframe.

    Returns:
        pd.DataFrame: All cases in the time range
    """
    case_dataset = asyncio.run(
        create_case_dataset_for_time_range(start_month=start_month, end_month=end_month)
    )
    df = convert_case_dataset_to_df(case_dataset)
    df.to_csv(path)

    return df


def concatenate_case_data(dirname: Path, filename: Path) -> pd.DataFrame:
    """
    Takes all the case data files in a directory and concatenates them
    into one dataframe, deduplicating as necessary. It then saves the
    dataframe into that directory.

    Args:
        dirname (Path): Directory path.
        filename (Path): Filename for concatenated data.

    Returns:
        pd.DataFrame: Concatenated data.
    """
    dfs = []

    for file_path in dirname.glob("*.csv"):
        df = pd.read_csv(file_path, index_col=0)
        dfs.append(df)

    concat_df = pd.concat(dfs, ignore_index=True)
    final_df = concat_df.drop_duplicates().reset_index(drop=True)
    final_df.to_csv(filename)

    return final_df


class Decision(Enum):
    DENIED = auto()
    GRANTED = auto()
    DISMISSED = auto()
    WITHDRAWN = auto()
    UNCLEAR = auto()


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data so that decisions and judges can be grouped and aggregated.

    Args:
        df (pd.DataFrame): Raw case dataframe.

    Returns:
        pd.DataFrame: Cleaned case dataframe.
    """
    # Clean judge names
    df["judge"] = df["judge"].str.split(",").str[0]
    df["judge"] = df["judge"].replace("Smith", "R.S. Smith")

    # Clean decision
    def get_decision(raw_decision: str) -> Decision:
        match raw_decision:
            case _ if "denied" in raw_decision or "denid" in raw_decision:
                return Decision["DENIED"]
            case _ if "granted" in raw_decision:
                return Decision["GRANTED"]
            case _ if "dismissed" in raw_decision:
                return Decision["DISMISSED"]
            case _ if "withdrawn" in raw_decision:
                return Decision["WITHDRAWN"]
            case _:
                return Decision["UNCLEAR"]

    df["decision"] = df["decision"].str.lower().apply(get_decision)
    return df


def get_case_stats_by_judge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get case stats and other info by judge.

    Args:
        df (pd.DataFrame): Dataframe of case-by-case data.

    Returns:
        pd.DataFrame: Aggregate data per judge.
    """
    # Construct pivot table for decision stats
    df["decision"] = df["decision"].apply(lambda x: x.name)
    pivot_table = df.pivot_table(
        index="judge", columns="decision", aggfunc="size", fill_value=0
    )
    pivot_table.loc["Average"] = pivot_table.sum(axis=0)
    decision_stats = pivot_table.div(pivot_table.sum(axis=1), axis=0)

    # Clean up decision stats and add some convenient sums
    decision_stats = decision_stats.rename(columns=lambda c: "PCT_" + c)
    decision_stats["PCT_DENIED_OR_DISMISSED"] = (
        decision_stats["PCT_DENIED"] + decision_stats["PCT_DISMISSED"]
    )

    # Calculate non-decision related stats
    other_stats = df.groupby("judge").agg(
        TOTAL_CASES=pd.NamedAgg(column="link", aggfunc="count"),
        FIRST_CASE_DATE=pd.NamedAgg(column="date", aggfunc="min"),
        LAST_CASE_DATE=pd.NamedAgg(column="date", aggfunc="max"),
    )

    # Concatenate all stats and return
    stats = pd.concat([decision_stats, other_stats], axis=1)
    return stats


def get_file_name(start_month: Month, end_month: Month) -> str:
    """
    Given a start month and an end month, generate the appropriate standardized
    filename to save that data.

    Args:
        start_month (Month): Start month.
        end_month (Month): End month.

    Returns:
        str: Filename.
    """
    start_name = f"{start_month.month}-{start_month.year}"
    end_name = f"{end_month.month}-{end_month.year}"
    return f"{start_name}_to_{end_name}_cases.csv"


def main():
    # Make directories to store data if they don't exist
    DATA_DIR = "data"
    RAW_DIR = "raw"
    Path(DATA_DIR, RAW_DIR).mkdir(parents=True, exist_ok=True)

    # Get most recent cases
    start_month = Month(2, 2004)
    end_month = Month(5, 2024)
    get_all_cases(
        start_month=start_month,
        end_month=end_month,
        path=Path(DATA_DIR, RAW_DIR, get_file_name(start_month, end_month)),
    )

    # Concatenate, dedupe, clean, and save all cases from raw data
    all_cases = concatenate_case_data(
        dirname=Path(DATA_DIR, RAW_DIR), filename=Path(DATA_DIR, "all_cases.csv")
    )
    clean_cases = clean_df(all_cases)
    clean_cases.to_csv(Path(DATA_DIR, "clean_cases.csv"))

    # Create stats by judge from clean case data
    stats = get_case_stats_by_judge(clean_cases)
    stats.to_csv(Path(DATA_DIR, "judge_stats.csv"))


if __name__ == "__main__":
    main()
