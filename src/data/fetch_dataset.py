# -*- coding: utf-8 -*-
import logging
import os
import pathlib

import click


@click.command()
@click.argument('kaggle_competition')
def main(kaggle_competition: str) -> None:
    """Fetches data files from Kaggle.

    Retrieves all data files pertaining to the provided competition from the 
    Kaggle website.  Data will be saved in the ../data/raw directory.  
    All data competition files will be downloaded with originial filenames 
    intact.

    Args:
        kaggle_competition: The name of the kaggle competition.
    """
    project_dir = pathlib.Path(__file__).resolve().parents[2]
    raw_data_path = os.path.join(project_dir, 'data', 'raw')

    logger = logging.getLogger(__name__)
    logger.info('fetching raw data...')
    os.system('kaggle datasets download {0} -p {1}'.format(
        kaggle_competition, raw_data_path))
    logger.info('done')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()  # pylint: disable=no-value-for-parameter
