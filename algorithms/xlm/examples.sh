time head -1000 ../../datasets/mnist_train.csv | python run.py --pathtodataset ../../datasets/mnist_train.csv --numhiddenneurons 1000 --scalefactor 255 --fillmissingvalues 0 | python evaluate.py
time head -1000 ../../datasets/quartic_cleaned.csv | venv/bin/python run.py --pathtodataset ../../datasets/quartic_cleaned.csv --numhiddenneurons 1000 --fillmissing 1 | venv/bin/python evaluate.py