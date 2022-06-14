from psyki.logic.datalog.grammar.adapters.antlr4 import get_formula_from_string
from psyki.ski.injectors import LambdaLayer
from setuptools import setup, find_packages
import pathlib
import subprocess
import distutils.cmd
from tensorflow.python.framework.random_seed import set_seed
from resources.data import get_splice_junction_data, data_to_int, get_binary_data, \
    get_splice_junction_extended_feature_mapping
from resources.data.splice_junction import CLASS_MAPPING, AGGREGATE_FEATURE_MAPPING
from resources.execution.utils import k_fold_cross_validation, create_fully_connected_nn_with_dropout, split
from resources.rules import get_splice_junction_rules, get_splice_junction_datalog_rules, get_binary_datalog_rules


# current directory
here = pathlib.Path(__file__).parent.resolve()

version_file = here / 'VERSION'

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')


def format_git_describe_version(version):
    if '-' in version:
        splitted = version.split('-')
        tag = splitted[0]
        index = f"dev{splitted[1]}"
        return f"{tag}.{index}"
    else:
        return version


def get_version_from_git():
    try:
        process = subprocess.run(["git", "describe"], cwd=str(here), check=True, capture_output=True)
        version = process.stdout.decode('utf-8').strip()
        version = format_git_describe_version(version)
        with version_file.open('w') as f:
            f.write(version)
        return version
    except subprocess.CalledProcessError:
        if version_file.exists():
            return version_file.read_text().strip()
        else:
            return '0.1.0'


version = get_version_from_git()

print(f"Detected version {version} from git describe")


class GetVersionCommand(distutils.cmd.Command):
    """A custom command to get the current project version inferred from git describe."""

    description = 'gets the project version from git describe'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        print(version)


class RunKINS(distutils.cmd.Command):
    """A custom command to execute experiments using KINS algorithm."""

    fold = 'fold'
    split = 'split'
    description = 'generate a csv file reporting the performance of KINS on the spite-junction dataset'
    user_options = [('mode=', 'm', 'experiment mode:\n'
                                   ' - fold (10-fold cross validation, default)\n'
                                   ' - split (2/3 for training, 1/3 for testing)'),
                    ('seed=', 's', 'starting seed, default is 0'),
                    ('file=', 'f', 'result file name, default "result" (.csv)')]

    def initialize_options(self):
        self.mode = self.fold
        self.seed = 0
        self.file = 'result'

    def finalize_options(self):
        self.seed = int(self.seed)

    def run(self):
        set_seed(self.seed)
        # Loading dataset and apply one-hot encoding for each feature
        # This means that for feature i_th we have 4 new features, one for each base.
        data = get_splice_junction_data('data')
        y = data_to_int(data.iloc[:, -1:], CLASS_MAPPING)
        x = get_binary_data(data.iloc[:, :-1], AGGREGATE_FEATURE_MAPPING)
        y.columns = [x.shape[1]]
        data = x.join(y)
        # Loading rules and conversion in Datalog form
        rules = get_splice_junction_rules('kb')
        rules = get_splice_junction_datalog_rules(rules)
        rules = get_binary_datalog_rules(rules)
        rules = [get_formula_from_string(rule) for rule in rules]
        # Creation of the base model
        model = create_fully_connected_nn_with_dropout()
        injector = LambdaLayer(model, CLASS_MAPPING, get_splice_junction_extended_feature_mapping())
        if self.mode == self.fold:
            result = k_fold_cross_validation(data, injector, rules, seed=self.seed)
        elif self.mode == self.split:
            result = split(data, injector, rules, seed=self.seed)
        else:
            raise Exception('Unexpected experiment mode')
        result.to_csv(self.file + '.csv', sep=';')

class RunKINS(distutils.cmd.Command):
    """A custom command to execute experiments using KINS algorithm."""

    fold = 'fold'
    split = 'split'
    description = 'generate a csv file reporting the performance of KINS on the spite-junction dataset'
    user_options = [('mode=', 'm', 'experiment mode:\n'
                                   ' - fold (10-fold cross validation, default)\n'
                                   ' - split (2/3 for training, 1/3 for testing)'),
                    ('seed=', 's', 'starting seed, default is 0'),
                    ('file=', 'f', 'result file name, default "result" (.csv)')]

    def initialize_options(self):
        self.mode = self.fold
        self.seed = 0
        self.file = 'result'

    def finalize_options(self):
        self.seed = int(self.seed)

    def run(self):
        set_seed(self.seed)
        # Loading dataset and apply one-hot encoding for each feature
        # This means that for feature i_th we have 4 new features, one for each base.
        data = get_splice_junction_data('data')
        y = data_to_int(data.iloc[:, -1:], CLASS_MAPPING)
        x = get_binary_data(data.iloc[:, :-1], AGGREGATE_FEATURE_MAPPING)
        y.columns = [x.shape[1]]
        data = x.join(y)
        # Loading rules and conversion in Datalog form
        rules = get_splice_junction_rules('kb')
        rules = get_splice_junction_datalog_rules(rules)
        rules = get_binary_datalog_rules(rules)
        rules = [get_formula_from_string(rule) for rule in rules]
        # Creation of the base model
        model = create_fully_connected_nn_with_dropout()
        injector = LambdaLayer(model, CLASS_MAPPING, get_splice_junction_extended_feature_mapping(), 0.2)
        if self.mode == self.fold:
            result = k_fold_cross_validation(data, injector, rules, seed=self.seed)
        elif self.mode == self.split:
            result = split(data, injector, rules, seed=self.seed)
        else:
            raise Exception('Unexpected experiment mode')
        result.to_csv(self.file + '.csv', sep=';')


setup(
    name='kins',  # Required
    version=version,
    description='KINS knowledge injection algorithm test',
    license='Apache 2.0 License',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/MatteoMagnini/kins-experiments',
    author='Matteo Magnini',
    author_email='matteo.magnini@unibo.it',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Prolog'
    ],
    keywords='symbolic knowledge injection, ski, symbolic ai',  # Optional
    # package_dir={'': 'src'},  # Optional
    packages=find_packages(),  # Required
    include_package_data=True,
    python_requires='>=3.9.0, <3.10',
    install_requires=[
        'psyki>=0.1.10',
        'tensorflow>=2.7.0',
        'numpy>=1.22.3',
        'scikit-learn>=1.0.2',
        'pandas>=1.4.2',
    ],  # Optional
    zip_safe=False,
    platforms="Independant",
    cmdclass={
        'get_project_version': GetVersionCommand,
        'run_kins_experiment': RunKINS,
    },
)
