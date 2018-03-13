import argparse
import random
import os


parser = argparse.ArgumentParser()
parser.add_argument('-i','--idir', help='directory from save files')
parser.add_argument('-o','--odir', help='directory to save files')
parser.add_argument('-q','--qty', help='quantity of test examples')

args = parser.parse_args()

if not args.idir:
    raise ValueError("missing argument --idir")

if not args.odir:
    raise ValueError("missing argument --odir")

if not args.qty:
    raise ValueError("missing argument --qty")

if args.idir:
    dir_in = args.idir

if args.odir:
    dir_out = args.odir

if args.qty:
    quantity = int(args.qty)

names_array = [] 
for filename in os.listdir(dir_in):
    name = filename.split(".")[0]
    if name not in names_array:
        names_array.append(name)

random_names = random.sample(names_array, quantity)
for name in random_names:
    print(name)
    os.rename(dir_in+name+".png", dir_out+name+".png")
    os.rename(dir_in+name+".png.txt", dir_out+name+".png.txt")

