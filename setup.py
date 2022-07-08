import glob
import os
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.pyplot import boxplot
from numpy import mean, array, arange
from pandas import read_csv
from psyki.logic.datalog.grammar.adapters import antlr4
from psyki.ski.injectors import LambdaLayer
from setuptools import setup, find_packages
import pathlib
import subprocess
import distutils.cmd
from tensorflow.python.framework.random_seed import set_seed

import resources.execution
from resources.data import get_dataset
from resources.results import PATH, single_class_accuracies, macro_f1, accuracy, weighted_f1
from resources.execution import create_standard_fully_connected_nn, run_experiments
from resources.rules import get_rules, FEATURE_MAPPING as POKER_FEATURE_MAPPING, CLASS_MAPPING as POKER_CLASS_MAPPING

# current directory
here = pathlib.Path(__file__).parent.resolve()

version_file = here / 'VERSION'

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')


def format_git_describe_version(version):
    if '-' in version:
        splitted = resources.execution.run_experiments('-')
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

    seed = 42
    epochs = 100
    yes = 'y'
    yes_matches = ('y', 'Y', 'yes', 'Yes', 'YES')
    description = 'generate a csv file reporting the performance of KILL on the poker hands dataset'
    user_options = [('early=', 'e', 'early stop conditions: [y]/n'),
                    ('knowledge=', 'k', 'knowledge during training: [y]/n')]

    def initialize_options(self):
        self.file = 'result'
        self.knowledge = self.yes
        self.early = self.yes

    def finalize_options(self):
        self.knowledge = self.knowledge in self.yes_matches
        self.early = self.early in self.yes_matches

    def run(self):
        set_seed(self.seed)
        data = pd.DataFrame(get_dataset('train'), dtype='int32')
        test = pd.DataFrame(get_dataset('test'), dtype='int32')
        formulae = [antlr4.get_formula_from_string(rule) for rule in get_rules()]
        model = create_standard_fully_connected_nn(10, 10, 3, 128, 'relu')

        injector = LambdaLayer(model, POKER_CLASS_MAPPING, POKER_FEATURE_MAPPING)
        result = run_experiments(data, injector, formulae, test=test, seed=self.seed, epochs=self.epochs,
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
        folder = str(PATH / self.folder)

        print('\n\nComputing statistics for ' + self.folder + ' experiments.\n')

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

        print('Accuracy: ' + str(mean(acc)))
        print("Macro F1: " + str(mean(mf1)))
        print("Weighted F1: " + str(mean(wf1)))
        print("Single class accuracies:")
        print(mean(array(single_class_acc), axis=0))

        print("\n\n" + 50 * '-' + '\n\n')

        print("Generating class accuracy distributions plot")
        x_labels = ['Nothing', 'Pair', 'Two Pairs', 'Three', 'Straight',
                    'Flush', 'Full', 'Four', 'Straight F.', 'Royal F.']
        plt.figure(figsize=(20, 10))
        main_color = 'blue'
        border_color = 'royalblue'
        box1 = boxplot(array(single_class_acc), positions=list(range(1, 11, 1)),
                notch=False, patch_artist=True,
                boxprops=dict(facecolor=border_color, color=main_color),
                capprops=dict(color=main_color),
                whiskerprops=dict(color=main_color),
                flierprops=dict(color=main_color, markeredgecolor=main_color),
                medianprops=dict(color=main_color),)

        other_folder = [x for x in self.allowed_folders if x != self.folder][0]
        other_folder_path = str(PATH / other_folder)
        confusion_matrices = []
        for file in glob.glob(other_folder_path + '/' + self.base_confusion_matrix_file_name + '*.csv'):
            confusion_matrices.append(read_csv(file).iloc[:, 1:].to_numpy())

        single_class_acc = []
        for cm in confusion_matrices:
            single_class_acc.append(single_class_accuracies(cm))
        main_color = 'red'
        border_color = 'pink'
        box2 = plt.boxplot(array(single_class_acc), positions=list(arange(1.5, 11.5, 1)),
                           notch=False, patch_artist=True)
        for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
            plt.setp(box2[item], color=main_color)
        plt.setp(box2["boxes"], facecolor=border_color)
        plt.setp(box2["fliers"], markeredgecolor=main_color)
        plt.xticks(list(range(1, 11)), list(range(1, 11)))
        plt.xlabel('Classes', fontsize=20)
        plt.ylabel('Accuracy', fontsize=20)
        plt.title('Class accuracy distributions', fontsize=32)
        # plt.subplot().set_xticklabels(list(POKER_CLASS_MAPPING.keys()), fontsize=16)
        plt.subplot().set_xticklabels(x_labels, fontsize=16)
        plt.subplot().legend([box1['boxes'][0], box2['boxes'][0]], [self.folder, other_folder], fontsize=20)
        # plt.show()
        plt.savefig(PATH / 'class-accuracy-distributions.pdf', format='pdf', bbox_inches='tight')
        print("Plot available at " + str(PATH / 'class-accuracy-distributions.pdf'))


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
