import csv
from io import StringIO

file_path = 'RGB0521_DL5_Fig5_MWLSED_StateD.ecsv'

with open(file_path, 'r') as f:
    lines = f.readlines()

header_lines = []
data_lines = []
in_data = False

for line in lines:
    if line.startswith('#'):
        header_lines.append(line)
    elif not in_data:
        # column names line
        header_lines.append(line.replace(' ', ','))
        in_data = True
    else:
        data_lines.append(line.strip())

# now process data_lines
output = StringIO()
writer = csv.writer(output, delimiter=',', quotechar='"', quoting=csv.QUOTE_NONNUMERIC)

for line in data_lines:
    reader = csv.reader(StringIO(line), delimiter=' ', quotechar='"')
    try:
        row = next(reader)
        writer.writerow(row)
    except StopIteration:
        pass

data_output = output.getvalue()

# write back
with open(file_path, 'w') as f:
    f.writelines(header_lines)
    f.write(data_output)