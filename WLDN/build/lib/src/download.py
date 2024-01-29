import os
import subprocess
import urllib.request
import pandas as pd
from tqdm import tqdm

"""
Adapted from Mike Fortunato at:
https://gitlab.com/mlpds_mit/ASKCOS/template-relevance/-/blob/dev/temprel/data/download.py
"""

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def get_uspto_480k():
    if not os.path.exists('data'):
        os.mkdir('data')
    os.chdir('data')
    print('DOWNLOAD 1 OF 3')
    download_url(
        'https://github.com/connorcoley/rexgen_direct/raw/master/rexgen_direct/data/train.txt.proc.tar.gz',
        'train.txt.tar.gz'
    )
    print('DECOMPRESSING 1 OF 3')
    subprocess.run(['tar', 'zxf', 'train.txt.tar.gz'])

    print('DOWNLOAD 2 OF 3')
    download_url(
        'https://github.com/connorcoley/rexgen_direct/raw/master/rexgen_direct/data/valid.txt.proc.tar.gz',
        'valid.txt.tar.gz'
    )
    print('DECOMPRESSING 2 OF 3')
    subprocess.run(['tar', 'zxf', 'valid.txt.tar.gz'])

    print('DOWNLOAD 3 OF 3')
    download_url(
        'https://github.com/connorcoley/rexgen_direct/raw/master/rexgen_direct/data/test.txt.proc.tar.gz',
        'test.txt.tar.gz'
    )
    print('DECOMPRESSING 3 OF 3')
    subprocess.run(['tar', 'zxf', 'test.txt.tar.gz'])
