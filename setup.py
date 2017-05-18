# 一般用setuptools
from setuptools import setup, find_packages,Command
# 维持不同平台文件相同的编码
from codecs import open
import distutils
from os import path
import os
import subprocess


here = path.abspath(path.dirname(__file__))

# 用同文件夹下的README.rst文件定义长介绍
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

# 用同文件夹下的requirements.txt文件定义运行依赖
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    REQUIREMETS = f.readlines()

packages=find_packages(exclude=['contrib', 'docs', 'test'])

class CoverageCommand(Command):
    description = "覆盖率"
    user_options = [
    ("output=","o","选择报告的输出方式")
    ]
    def initialize_options(self):
        self.cwd = None
        self.output = ''
    def finalize_options(self):
        self.cwd = os.getcwd()
        if self.output and self.output not in ("report","html"):
            raise Exception("Parameter --output is missing")
    def run(self):
        assert os.getcwd() == self.cwd, 'Must be in package root: {self.cwd}'.format(self=self)
        command = ['/usr/bin/env', 'python', '-m', 'coverage']
        if self.output:
            command.append('{self.output}'.format(self=self))
        else:
            command.append('report')
        self.announce('Running command: {command}'.format(command = str(command)),
            level=distutils.log.INFO)
        subprocess.check_call(command)


class TestCommand(Command):
    description = "测试"
    user_options = []
    def initialize_options(self):
        self.cwd = None
    def finalize_options(self):
        self.cwd = os.getcwd()
    def run(self):
        assert os.getcwd() == self.cwd, 'Must be in package root: {self.cwd}'.format(self=self)
        command = ['/usr/bin/env', 'python', '-m',
        'coverage','run' ,'--source=score_card_model',
        '-m', 'unittest', 'discover', '-v', '-s', 'test']
        self.announce('Running command: {command}'.format(command = str(command)),
            level=distutils.log.INFO)
        subprocess.check_call(command)

setup(
    name='score_card_model',
    version='0.0.1',
    description='A sample Python project',
    long_description=long_description,

    # 项目地址
    url='https://github.com/pypa/sampleproject',

    # 作者信息
    author='The Python Packaging Authority',
    author_email='pypa-dev@googlegroups.com',
    # 维护者信息
    maintainer = "",
    maintainer_email = "",
    # 指定可用的平台,一般有c扩展的可能会用到
    platforms = ["any"],

    # 许可证信息
    license='MIT',

    # 分类信息,具体看 https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # 发展时期,常见的如下
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # 开发的目标用户
        'Intended Audience :: Developers',
        # 属于什么类型
        'Topic :: Software Development :: Build Tools',

        # 许可证信息
        'License :: OSI Approved :: MIT License',

        # 目标python版本
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],

    # 关键字
    keywords='sample setuptools development',

    # 指定用到的模块,find_packages会找到同文件夹下的模块,用`exclude`指定排除的模块
    packages=packages,

    # 运行时使用的依赖
    install_requires=REQUIREMETS,
    # 是否支持直接引用zip文件,这是setuptools的特有功能
    zip_safe=False,

    # 额外环境的依赖,一般不单独用文件指出
    # for example:
    # pip install -e .[dev,test]
    # extras_require={
    #     'dev': ['check-manifest'],
    #     'test': ['coverage'],
    # },
    # 指定可执行脚本,如果安装,脚本会被放到默认安装路径
    #scripts=["scripts/test.py"],

    # 模块如果有自带的数据文件,可以用package_data指定
    #package_data={
    #    'sample': ['package_data.dat'],
    #},

    # 指定模块自带数据所在的文件夹
    data_files=[('./', ['requirements.txt'])],
    # 定义自定义命令
    cmdclass = {
        'coverage':CoverageCommand,
        'test':TestCommand
    }
)
