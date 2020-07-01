import errno
import os
import csv


def str2bool(s):
    return s.lower() in ('true', '1')


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def save_to_csv(file, fields):
    with open(file, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)