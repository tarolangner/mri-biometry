import sys
import os
import numpy as np

def main(argv):

    ids_path_file = "ids_cv0.txt" # subject ids as rows
    csv_path_file = "path_to_csv.txt" # path to UK Biobank style csv, single column

    path_out = "output/"

    (ids, values) = selectValidEntries(ids_path_file, csv_path_file)

    if not os.path.exists(path_out): os.makedirs(path_out)

    writeOutput(path_out, ids, values)


def writeOutput(path_out, ids, values):

    with open(path_out + "gt.txt", "w") as f:
        f.write("eid,gt\n")
        for i in range(len(ids)):
            f.write("{},{}\n".format(ids[i], values[i]))
        

def selectValidEntries(ids_path_file, csv_path_file):

    # Read ids
    with open(ids_path_file) as f: ids = f.readlines()
    ids = np.array(ids).astype("int")

    # Read csv
    with open(csv_path_file) as f:
        csv_path = f.readlines()[0].replace("\n","")

    #
    with open(csv_path) as f: entries = f.readlines()
    entries.pop(0) # remove csv header

    ids_csv = [f.split(",")[0].replace("\"","") for f in entries]
    values_csv = [f.split(",")[1].replace("\"","").replace("\n","") for f in entries]

    ids_csv = np.array(ids_csv).astype("int")
    values_csv = np.array(values_csv).astype("float")

    print("{} subjects selected".format(len(ids)))

    # Mask out non-selected
    mask = np.in1d(ids_csv, ids)
    ids_out = ids_csv[mask]
    values_out = values_csv[mask]

    print("{} entries found".format(len(ids_out)))

    # Mask out invalid label values
    mask = np.invert(np.isnan(values_out))
    ids_out = ids_out[mask]
    values_out = values_out[mask]

    print("{} with valid entries".format(len(ids_out)))

    return (ids_out, values_out)


if __name__ == '__main__':
    main(sys.argv)
