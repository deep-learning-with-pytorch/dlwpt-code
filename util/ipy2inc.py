import argparse
import json
import sys

import logging
log = logging.getLogger(__name__)
# log.setLevel(logging.WARN)
log.setLevel(logging.INFO)
# log.setLevel(logging.DEBUG)


class NotebookToTextApp(object):
    @classmethod
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('input_path',
            help='.ipynb file to parse',
            # default=256,
            # type=int,
        )
        # parser.add_argument('--num-workers',
        #     help='Number of worker processes for background data loading',
        #     default=8,
        #     type=int,
        # )
        # parser.add_argument('--scaled',
        #     help="Scale the CT chunks to square voxels.",
        #     default=False,
        #     action='store_true',
        # )

        self.cli_args = parser.parse_args(sys_argv)

    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.cli_args))

        with open(self.cli_args.input_path) as nb_file:
            nb = json.load(nb_file)

        with open(self.cli_args.input_path.replace('.ipynb', '.nbinclude'), 'w', encoding='utf8') as out_file:
            for cell_index, cell_dict in enumerate(nb['cells']):
                if cell_dict['cell_type'] != 'code':
                    continue

                cell_index += 1

                print('# tag::cell_{}'.format(cell_index), file=out_file)
                print('# In[{}]:'.format(cell_index), file=out_file)
                print('# tag::cell_{}_code'.format(cell_index), file=out_file)
                print(''.join(cell_dict['source']), file=out_file)
                print('# end::cell_{}_code'.format(cell_index), file=out_file)
                print('', file=out_file)

                print('# Out[{}]:'.format(cell_index), file=out_file)
                print('# tag::cell_{}_output'.format(cell_index), file=out_file)

                for output_dict in cell_dict['outputs']:
                    output_list = output_dict.get('data', {}).get('text/plain', [])
                    if output_list:
                        print(''.join(output_list), file=out_file)

                print('# end::cell_{}_output'.format(cell_index), file=out_file)
                print('# end::cell_{}'.format(cell_index), file=out_file)
                print('\n', file=out_file)


if __name__ == '__main__':
    sys.exit(NotebookToTextApp().main() or 0)
