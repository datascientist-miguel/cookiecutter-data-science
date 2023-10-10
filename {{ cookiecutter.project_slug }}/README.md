# "Project Data Science {{ cookiecutter.project_slug }}"

By: {{ cookiecutter.project_author_name }}.

contact: {{ cookiecutter.project_author_email }}

Version: {{ cookiecutter.project_version }}

{{ cookiecutter.project_description }}

## Prerequisites

- [Anaconda](https://www.anaconda.com/download/) >=4.x

## Create and activate environment

The environment is created automatically, so you only need to activate it.
verify that the environment is created with the following command:

```bash
conda env list
```
to activate the environment:

```bash
conda activate {{ cookiecutter.project_slug }}
```
## Licence

{% if cookiecutter.open_source_license == 'MIT' %} This project has The MIT License (MIT) Copyright (c) {% now 'utc', '%Y' %}, {{ cookiecutter.project_author_name }}

{% elif cookiecutter.open_source_license == 'BSD-3-Clause' %} Copyright (c) {% now 'utc', '%Y' %}, {{ cookiecutter.project_author_name }} All rights reserved.

{% elif cookiecutter.open_source_license == 'No license file' %} This project has not a license file {% endif %}


## Project organization

{{ cookiecutter.project_slug }}
├── LICENCE
├── README.md
├── environment.yml
├── .gitignore
├── data
│   ├── raw
│   └── processed
├── docs
├── models
├── notebooks
│   └── 0.0-{{ cookiecutter.project_slug }}-introduction.ipynb
|   └── 0.1-{{ cookiecutter.project_slug }}-eda.ipynb
|   └── 0.2-{{ cookiecutter.project_slug }}-visualization.ipynb
|   └── 0.3-{{ cookiecutter.project_slug }}-modeling.ipynb
├── report
├── scripts
│   ├── analysis
│   ├── manipulation
│   ├── modeling
│   ├── visualization