from setuptools import setup, find_packages

setup(name='clip_gpt',
      version='0.0',
      description='161M parameter CLIP GPT model initialized with OpenAI CLIP weights',
      url='',
      author='Noa Nabeshima',
      author_email='noanabeshima@gmail.com',
      license='MIT',
      packages=find_packages("."),
      zip_safe=False)