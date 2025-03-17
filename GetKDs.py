import pandas as pd
import sys

if __name__ == '__main__':
    all_kds = []
    for filename in sys.argv[1:]:
        df = list(pd.read_excel(filename,sheet_name=None,engine='openpyxl').values())[0]
        Kds = df['Kd (mM)'].to_list()
        all_kds = all_kds + Kds
    #
    print(all_kds)