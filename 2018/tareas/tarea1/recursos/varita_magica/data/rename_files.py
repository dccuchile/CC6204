import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-d','--dir', help='directory to rename files')

args = parser.parse_args()

if not args.dir:
    raise ValueError("missing argument --dir")

if args.dir:
    dir_in = args.dir


dict_name = dict()
idx = 1
for filename in os.listdir(dir_in):
    name = filename.split(".")[0]
    if name not in dict_name:
        dict_name[name] = "{0:03}".format(idx)
        idx += 1

for filename in os.listdir(dir_in):
    extention = filename.split(".png")[1]
    for name, idx in dict_name.items():
        if filename.startswith(name):
            if extention is "":
                os.rename(dir_in + filename, dir_in + idx + ".png")
                os.rename(dir_in + filename + ".txt", dir_in + idx + ".txt")