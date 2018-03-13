#!/bin/sh

mkdir test_set
mkdir test_set/hechizo-0/
mkdir test_set/hechizo-1/
mkdir test_set/hechizo-2/
mkdir test_set/hechizo-3/
mkdir test_set/hechizo-4/
mkdir test_set/hechizo-5/
mkdir test_set/hechizo-6/
mkdir test_set/hechizo-7/
mkdir test_set/hechizo-8/
mkdir test_set/hechizo-9/

mkdir train_set
mkdir train_set/hechizo-0/
mkdir train_set/hechizo-1/
mkdir train_set/hechizo-2/
mkdir train_set/hechizo-3/
mkdir train_set/hechizo-4/
mkdir train_set/hechizo-5/
mkdir train_set/hechizo-6/
mkdir train_set/hechizo-7/
mkdir train_set/hechizo-8/
mkdir train_set/hechizo-9/

python split_train_test.py --idir ./hechizo-0/ --odir ./test_set/hechizo-0/ --qty 50
python split_train_test.py --idir ./hechizo-1/ --odir ./test_set/hechizo-1/ --qty 50
python split_train_test.py --idir ./hechizo-2/ --odir ./test_set/hechizo-2/ --qty 50
python split_train_test.py --idir ./hechizo-3/ --odir ./test_set/hechizo-3/ --qty 50
python split_train_test.py --idir ./hechizo-4/ --odir ./test_set/hechizo-4/ --qty 50
python split_train_test.py --idir ./hechizo-5/ --odir ./test_set/hechizo-5/ --qty 50
python split_train_test.py --idir ./hechizo-6/ --odir ./test_set/hechizo-6/ --qty 50
python split_train_test.py --idir ./hechizo-7/ --odir ./test_set/hechizo-7/ --qty 50
python split_train_test.py --idir ./hechizo-8/ --odir ./test_set/hechizo-8/ --qty 50
python split_train_test.py --idir ./hechizo-9/ --odir ./test_set/hechizo-9/ --qty 50

python rename_files.py --dir ./hechizo-0/
python rename_files.py --dir ./hechizo-1/
python rename_files.py --dir ./hechizo-2/
python rename_files.py --dir ./hechizo-3/
python rename_files.py --dir ./hechizo-4/
python rename_files.py --dir ./hechizo-5/
python rename_files.py --dir ./hechizo-6/
python rename_files.py --dir ./hechizo-7/
python rename_files.py --dir ./hechizo-8/
python rename_files.py --dir ./hechizo-9/
python rename_files.py --dir ./test_set/hechizo-0/
python rename_files.py --dir ./test_set/hechizo-1/
python rename_files.py --dir ./test_set/hechizo-2/
python rename_files.py --dir ./test_set/hechizo-3/
python rename_files.py --dir ./test_set/hechizo-4/
python rename_files.py --dir ./test_set/hechizo-5/
python rename_files.py --dir ./test_set/hechizo-6/
python rename_files.py --dir ./test_set/hechizo-7/
python rename_files.py --dir ./test_set/hechizo-8/
python rename_files.py --dir ./test_set/hechizo-9/

mv ./hechizo-0/ train_set/
mv ./hechizo-1/ train_set/
mv ./hechizo-2/ train_set/
mv ./hechizo-3/ train_set/
mv ./hechizo-4/ train_set/
mv ./hechizo-5/ train_set/
mv ./hechizo-6/ train_set/
mv ./hechizo-7/ train_set/
mv ./hechizo-8/ train_set/
mv ./hechizo-9/ train_set/