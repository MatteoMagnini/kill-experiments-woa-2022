import glob
import os
import pandas as pd
from numpy import mean, ndarray, array
from pandas import read_csv
from psyki.logic.datalog.grammar.adapters import antlr4
from psyki.ski.injectors import LambdaLayer
from setuptools import setup, find_packages
import pathlib
import subprocess
import distutils.cmd
from tensorflow.python.framework.random_seed import set_seed
from resources.data import get_dataset
from resources.results import PATH, sum_confusion_matrix, single_class_accuracies, macro_f1, accuracy, weighted_f1
from resources.rules.poker import CLASS_MAPPING as POKER_CLASS_MAPPING, FEATURE_MAPPING as POKER_FEATURE_MAPPING
from resources.execution.utils import run_experiments, create_standard_fully_connected_nn
from resources.rules import get_rules


# current directory
here = pathlib.Path(__file__).parent.resolve()

version_file = here / 'VERSION'

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')


def format_git_describe_version(version):
    if '-' in version:
        splitted = version.run_experiments('-')
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


class RunKILLExperiments(distutils.cmd.Command):
    """A custom command to execute experiments using KINS algorithm."""

    gamma = None
    yes = 'y'
    yes_matches = ('y', 'Y', 'yes', 'Yes', 'YES')
    sj = 'splice-junction'
    description = 'generate a csv file reporting the performance of KINS on the spite-junction dataset'
    user_options = [('early=', 'e', 'early stop conditions: [y]/n'),
                    ('seed=', 's', 'starting seed, default is 0'),
                    ('knowledge=', 'k', 'knowledge during training: [y]/n'),
                    ('file=', 'f', 'result file name, default "result" (.csv)')]

    def initialize_options(self):
        self.seed = 42
        self.file = 'result'
        self.knowledge = self.yes
        self.early = self.yes

    def finalize_options(self):
        self.seed = int(self.seed)
        self.knowledge = self.knowledge in self.yes_matches
        self.early = self.early in self.yes_matches

    def run(self):
        set_seed(self.seed)
        # Loading dataset and apply one-hot encoding for each feature
        # This means that for feature i_th we have 4 new features, one for each base.
        data = pd.DataFrame(get_dataset('train'), dtype='int32')
        test = pd.DataFrame(get_dataset('test'), dtype='int32')
        formulae = [antlr4.get_formula_from_string(rule) for rule in get_rules()]
        model = create_standard_fully_connected_nn(10, 10, 3, 128, 'relu')
        injector = LambdaLayer(model, POKER_CLASS_MAPPING, POKER_FEATURE_MAPPING, self.gamma)
        result = run_experiments(data, injector, formulae, test=test, seed=self.seed, training_ratio=900, epochs=100,
                                 use_knowledge=self.knowledge, stop=self.early)
        result.to_csv(self.file + '.csv', sep=';')


class ComputeStatistics(distutils.cmd.Command):
    """A custom command to get some statistics."""

    description = 'gets statistics on the same family of experiment'
    user_options = [('folder=', 'f', 'folder name where the results are stored, default "knowledge" else "classic"'), ]
    result_file_name = 'result'
    base_confusion_matrix_file_name = 'confusion_matrix'
    allowed_folders = ('knowledge', 'classic')

    def initialize_options(self):
        self.folder = 'knowledge'

    def finalize_options(self):
        self.folder = self.folder
        if self.folder not in self.allowed_folders:
            raise Exception('Unexpected folder name: ' + self.folder +
                            '\nAccepted values are: ' + ','.join(self.allowed_folders))

    def run(self):
        folder = str(PATH / self.folder)  # + '-old'
        num_files = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])
        if num_files == 0:
            raise Exception('No file found in ' + folder)
        confusion_matrices = []
        for file in glob.glob(folder + '/' + self.base_confusion_matrix_file_name + '*.csv'):
            confusion_matrices.append(read_csv(file).iloc[:, 1:].to_numpy())

        single_class_acc, acc, mf1, wf1 = [], [], [], []
        for cm in confusion_matrices:
            single_class_acc.append(single_class_accuracies(cm))
            acc.append(accuracy(cm))
            mf1.append(macro_f1(cm))
            wf1.append(weighted_f1(cm))

        print("Single class accuracies:")
        print(mean(array(single_class_acc), axis=0))
        print('Accuracy: ' + str(mean(acc)))
        print("Macro F1: " + str(mean(mf1)))
        print("Weighted F1: " + str(mean(wf1)))
        print("\n\n" + 50 * '-' + '\n\n')

        sum_cm = sum_confusion_matrix(confusion_matrices)
        print("Single class accuracies:")
        print(single_class_accuracies(sum_cm))
        print('Accuracy: ' + str(accuracy(sum_cm)))
        print("Macro F1: " + str(macro_f1(sum_cm)))
        print("Weighted F1: " + str(weighted_f1(sum_cm)))


setup(
    name='kill',  # Required
    version=version,
    description='KILL knowledge injection algorithm test',
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
        'run_kill_experiment': RunKILLExperiments,
        'get_statistics': ComputeStatistics,
    },
)
