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
        ├── data
        │   ├── processed      <- The final, canonical data sets for modeling.
        │   └── raw            <- The original, immutable data dump.
        │
        ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
        │                         the creator's initials, and a short `-` delimited description, e.g.
        │                         `1.0-jvelezmagic-initial-data-exploration`.
        │
        ├── .gitignore         <- Files to ignore by `git`.
        │
        ├── environment.yml    <- The requirements file for reproducing the analysis environment.
        │
        └── README.md          <- The top-level README for developers using this project.

