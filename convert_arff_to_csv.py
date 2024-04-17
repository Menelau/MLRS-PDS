import os

# Getting all the arff files from the S2 folder
files = [arff for arff in os.listdir('S2') if arff.endswith(".arff")]

# Function for converting arff list to csv list
def toCsv(content):
    data = False
    header = ""
    newContent = []
    for line in content:
        if not data:
            if "@attribute" in line:
                attri = line.split()
                columnName = attri[attri.index("@attribute")+1]
                header = header + columnName + ","
            elif "@data" in line:
                data = True
                header = header[:-1]
                header += '\n'
                newContent.append(header)
        else:
            newContent.append(line)
    return newContent

# Create the S2_csv folder if it doesn't exist
if not os.path.exists('S2_csv'):
    os.mkdir('S2_csv')

# Main loop for reading and writing files
for file in files:
    with open(os.path.join('S2', file) , "r") as inFile:
        content = inFile.readlines()
        name,ext = os.path.splitext(inFile.name)
        new = toCsv(content)
        csv_file = name.replace("-", "")+".csv"
        with open(os.path.join('S2_csv', csv_file), "w") as outFile:
            outFile.writelines(new)
