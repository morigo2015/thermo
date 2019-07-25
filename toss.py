
import os
import datetime
import time
import csv
from collections import namedtuple
import glob
import shutil

import flir_image_extractor

input_folder = '../data/input'  # 'input_data'
result_folder = '../data/result'
archive_folder = '../data/archive'
not_logged_folder = '../data/result/notlogged'

need_fir = True

def get_log( csv_fname: str) -> list:
    with open(csv_fname, 'r') as f:
      raw_log = list(csv.reader(f))
    del raw_log[0] # remove header

    # convert to namedtuple and convert timstamp string to datetime
    Row = namedtuple('Row', ['timestamp', 'comp_id', 'comp', 'equip_id', 'equip', 'view_id', 'view'])
    log = [ Row._make( [ datetime.datetime.strptime(r[0],'%Y-%m-%dT%H:%M:%S'), r[1],r[2],r[3],r[4],r[5],r[6] ])
            for r in raw_log
        ]
    return log

def copy_to_archive(dir_path):
    arc_folder = os.path.join(archive_folder, os.path.basename(dir_path))
    os.makedirs(arc_folder,exist_ok=True)
    for f in os.listdir(dir_path):
        src = os.path.join(dir_path,f)
        dst = os.path.join(arc_folder,os.path.basename(f))
        if not os.path.isfile(dst):
            shutil.copyfile(src,dst)
    print(f"all files from {dir_path} copied to {arc_folder}")

def get_all_logs(dir_path: str) -> list:
    log = []
    for csv_file in glob.glob(dir_path + '/' + '*.csv'):
        log += get_log(csv_file)
    log.sort(key=lambda tup: tup.timestamp)  # sort by timestamp, not to be surprised by input csv
    return log

def flirfname_2_dt(fname:str) -> datetime:
    fname = os.path.basename(fname)
    return datetime.datetime.strptime(fname[5:-4],'%Y%m%dT%H%M%S')

def get_log_entry(log, f_dt: datetime):
    for log_entry in reversed(log):
        if log_entry.timestamp <= f_dt:
            return log_entry
    return None

def get_new_fname(log_entry,file_timestamp) -> str:
    timestamp = datetime.datetime.strftime(file_timestamp,'%Y%m%dT%H%M%S')
    return f"{log_entry.comp_id}_{log_entry.equip_id}_{log_entry.view_id}_{timestamp}_.jpg"

def get_new_folder(log_entry, result_folder) -> str:
    new_folder = os.path.join(result_folder,log_entry.comp,log_entry.equip,log_entry.view)
    os.makedirs(new_folder,exist_ok=True)
    return new_folder

def main():

    fir = flir_image_extractor.FlirImageExtractor(image_suffix="v.jpg", thermal_suffix="t.png")
    for dir in next(os.walk(input_folder))[1]:
        dir_start_time = time.perf_counter()
        dir_file_counter = 0
        dir_path = os.path.join(input_folder,dir)

        log = get_all_logs(dir_path)
        print(f"folder:{dir_path} len(log)={len(log)}")
        for jpg_file in glob.glob(dir_path+'/'+'*.jpg'):
            fname_timestamp = flirfname_2_dt(jpg_file)
            log_entry = get_log_entry(log,fname_timestamp)
            if log_entry is not None:
                new_fname = get_new_fname(log_entry, fname_timestamp)
                new_folder = get_new_folder(log_entry,result_folder)
                new_fname_path = os.path.join(new_folder,new_fname)
                shutil.copyfile(jpg_file, new_fname_path )
                print(f"{jpg_file} ---> {new_folder}/{new_fname})")
                if need_fir:
                    fir.process_image(new_fname_path)
                    fir.save_images()
            else: # log entry isn't found
                folder = os.path.join(result_folder,not_logged_folder)
                os.makedirs(folder,exist_ok=True)
                os.rename(jpg_file, os.path.join(folder,os.path.basename(jpg_file)))
                print(f"{jpg_file} moved to {folder} since related log entry is not found")
            dir_file_counter += 1

        copy_to_archive(dir_path)
        shutil.rmtree(dir_path,ignore_errors=True)
        os.rmdir(dir_path)
        dt = time.perf_counter()-dir_start_time
        print(f"{dir_path}: processed {dir_file_counter} files in {dt/60.:.0f} minutes",
              f" avg speed = {dt/dir_file_counter:.0f} sec/file")

if __name__ == "__main__":
    main()

