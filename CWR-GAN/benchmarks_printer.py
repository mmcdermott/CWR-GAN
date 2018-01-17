# vim: set fileencoding=utf-8
from __future__ import print_function

def pad(c, ex='', p=2): return max(len(c), len(ex)) + p

# Benchmark Printing:
# TODO(mmd): figure out proper way of namsepacing these constants.
ITER_HEADER, TIME_HEADER, STEP_HEADER = 'Iteration', 'Δt (s)', 'Step Type'

class BenchmarkPrinter(object):
    """
    Prints benchmarks during training in a readable fashion.
    TODO(mmd): Proper docstring.
    TODO(mmd): STEP_HEADER params really aren't general...
    """
    def __init__(
        self,
        params   = {},
        columns         = [],
        default_columns = [ITER_HEADER, TIME_HEADER, STEP_HEADER],
        default_params  = {
            ITER_HEADER: (ITER_HEADER, '{}',      '1504785'),
            TIME_HEADER: (TIME_HEADER, '{:.2f}s', '20.32s'),
            STEP_HEADER: (STEP_HEADER, '{}',      'train, translator'),
        },
    ):
        self.benchmarks = default_columns + columns
        self.params     = params
        self.params.update(default_params)

        self.columns = [self.params[benchmark][0] for benchmark in self.benchmarks]
        self.fmts    = [self.params[benchmark][1] for benchmark in self.benchmarks]
        self.widths  = [pad(col,ex=self.params[bench][2]) for col,bench in zip(self.columns,self.benchmarks)]

    @classmethod
    def print(cls, data, widths, fmts=None, col_sep='|', upper='', lower=''):
        if fmts is None: fmts = ['{}'] * len(data)
        assert len(data) == len(widths)
        assert len(data) == len(fmts)

        formatted_data = map(lambda args: args[0].format(args[1]), zip(fmts, data))
        columns = map(lambda args: ('{: ^%i}' % args[0]).format(args[1]), zip(widths, formatted_data))
        row = col_sep + col_sep.join(columns) + col_sep

        total_width = sum(widths) + len(widths) + 1

        upper_bound = '' if upper == '' else upper * total_width + '\n'
        lower_bound = '' if lower == '' else '\n' + lower * total_width

        print(upper_bound + row + lower_bound)

    def pull(self, data): return [data[benchmark] for benchmark in self.benchmarks]
    def print_benchnames(self):
        BenchmarkPrinter.print(map(lambda s: s, self.columns),self.widths,upper='_',lower='-')
    def print_data_groups(self, data_groups):
        for data in data_groups[:-1]: BenchmarkPrinter.print(self.pull(data), self.widths, fmts=self.fmts)
        BenchmarkPrinter.print(self.pull(data_groups[-1]), self.widths, fmts=self.fmts, lower='-')
