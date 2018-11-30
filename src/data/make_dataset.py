# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
import logging
import os
import pathlib
from pathlib import Path

import click
import pandas as pd
from pandas.api import types


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
def main(input_filepath: str) -> None:
    """Runs data processing scripts. 
    
    Runs scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).

    Args:
        input_filepath: The filepath containing the raw data
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data...')
    df = pd.read_csv(input_filepath)
    df = process_data(df)
    logger.info('writing final data set to ./data/processed/out.csv')
    write_data(df)
    logger.info('done')


def _encode_ordinals(df: pd.DataFrame) -> pd.DataFrame:
    """Encodes ordinal features.

    Encodes catagorical ordinal features within the dataset as numeric features.

    Args:
        df: The dataframe to be encoded

    Returns:
        The encoded dataframe
    """
    subway_mapping = {
        '0-5min': 4,
        '5min~10min': 3,
        '10min~15min': 2,
        '15min~20min': 1,
        'no_bus_stop_nearby': 0
    }
    bus_mapping = {'0~5min': 2, '5min~10min': 1, '10min~15min': 0}
    df['TimeToSubway'] = df['TimeToSubway'].map(subway_mapping)
    df['TimeToBusStop'] = df['TimeToBusStop'].map(bus_mapping)
    return df


def process_data(df: pd.DataFrame) -> pd.DataFrame:
    """Processeses the data.

    Converts the catagorical features present in the dataset into numerical
    features for future analysis.
    
    Args:
        df: The dataframe to be converted

    Returns: 
        The processed dataframe with catagorical features removed.
    """
    # Mark catagorical features that must be one-hot encoded as catagorical.
    # Ensures get_dummies will work properly on test data
    df['HallwayType'] = df['HallwayType'].astype(
        types.CategoricalDtype(categories=['terraced', 'corridor', 'mixed']))
    df['HeatingType'] = df['HeatingType'].astype(
        types.CategoricalDtype(
            categories=['individual_heating', 'central_heating']))
    df['AptManageType'] = df['AptManageType'].astype(
        types.CategoricalDtype(
            categories=['management_in_trust', 'self_management']))
    df['SubwayStation'] = df['SubwayStation'].astype(
        types.CategoricalDtype(categories=[
            'Kyungbuk_uni_hospital', 'Daegu', 'Sin-nam', 'Myung-duk',
            'Chil-sung-market', 'Bangoge', 'Banwoldang', 'no_subway_nearby'
        ]))

    df = _encode_ordinals(df)

    df = pd.get_dummies(
        df,
        columns=[
            'HallwayType', 'HeatingType', 'AptManageType', 'SubwayStation'
        ])

    return df


def write_data(df: pd.DataFrame) -> None:
    """Writes dataframe to (../processed).
    
    Writes dataframe to a csv file named out.csv in the proccessed data 
    directory.

    Args:
        df: The dataframe to be written
    """
    project_dir = pathlib.Path(__file__).resolve().parents[2]
    processed_data_path = os.path.join(project_dir, 'data', 'processed')
    df.to_csv(os.path.join(processed_data_path, 'out.csv'))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()  # pylint: disable=no-value-for-parameter
