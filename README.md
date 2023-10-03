# Cookiecutter Project

## Requirements

- [Anaconda](https://www.anaconda.com/download/) >= 4.x
- [git](https://git-scm.com/)
- [Cookiecutter Python package](http://cookiecutter.readthedocs.org/en/latest/installation.html)

``` bash
conda install -c conda-forge cookiecutter
```

## Create a new project

In the directory where you want to create the project, run:

```bash
cookiecutter .
```

or

```bash
cookiecutter -c https://github.com/datascientist-miguel/cookiecutter-data-science

You will be asked about your project name, author, etc. This information will be used to create the project.


## Estructure of the project

├── LICENCE
├── README.md
├── data
│   ├── raw
│   └── processed
├── environment.yml
├── notebooks
│   └── 0.0-{{ cookiecutter.project_slug }}-introduction.ipynb
├── reports
├── scripts
├── setup.py
└── {{ cookiecutter.project_slug }}_packages
    ├── __init__.py
    ├── utils
    │   ├── __init__.py
    │   └── get_path.py
    └── visualization
        ├── __init__.py
        └── visualization.py

## Credits

This project was made it in base of:

Platzi Configuración Profesional de Entorno de Trabajo para Ciencia de Datos course by Jesús Vélez Santiago
Cookiecutter Data Science repository by Driven Data