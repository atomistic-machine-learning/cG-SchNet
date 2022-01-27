import argparse
import sys
from pathlib import Path
from ase.db import connect


def get_parser():
    """ Setup parser for command line arguments """
    main_parser = argparse.ArgumentParser()
    main_parser.add_argument('data_base_path', type=str,
                             help='Path to data base (.db file) with molecules.')

    return main_parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    
    path = Path(args.data_base_path).resolve()
    if not path.exists():
        print(f'Argument error! There is no data base at "{path}"!')
        sys.exit(0)

    with connect(args.data_base_path) as con:
        if con.count() == 0:
            print(f'Error: The data base at "{path}" is empty!')
            sys.exit(0)
        elif 'info' not in con.metadata:
            print(f'Error: The metadata of data base at "{path}" does not contain the field "info"!')
            print(f'However, this is the data stored for the first molecule:\n{con.get(1).data}')
            sys.exit(0)
        print(f'\nINFO for data base at "{path}"')
        print('===========================================================================================================')
        print(con.metadata['info'])
        print('===========================================================================================================')
        print('\nFor example, here is the data stored with the first three molecules:')
        for i in range(3):
            print(f'{i}: {con.get(i+1).data}')
        print('\nYou can load and access the molecules and accompanying data by connecting to the data base with ASE, e.g. using the following python code snippet:')
        print(f'from ase.db import connect')
        print(f'with connect({path}) as con:\n',
            f'\trow = con.get(1)  # load the first molecule, 1-based indexing\n',
            f'\tR = row.positions  # positions of atoms as 3d coordinates\n',
            f'\tZ = row.numbers  # list of atomic numbers\n',
            f'\tdata = row.data  # dictionary of data stored with the molecule\n')
        print(f'You can visualize the molecules in the data base with ASE from the command line by calling:')
        print(f'ase gui {path}')

